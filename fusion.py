"""
Gate Fusion for MPS quantum circuit simulation.

Gate fusion merges sequences of gates into a single combined gate before
simulation. This reduces the number of SVD truncations (for multi-qubit
gates) and einsum contractions (for single-qubit gates), often giving a
significant speedup.

Overlapping-support fusion
---------------------------
Fusion here is driven by **shared qubit support**, not by requiring gates
to sit on the exact same qubit or the exact same qubit pair. A run of
consecutive gates is fused into one block as long as each new gate's
qubits overlap the qubits already in the block — i.e. touch at least one
qubit the block already spans — and the block's total qubit count stays
within ``max_qubits``.

This means, for example, ``H(0)``, ``CNOT(0,1)``, ``Rz(1)`` fuse into a
single 2-qubit gate even though no two of them act on the identical qubit
set: ``H(0)`` and ``CNOT(0,1)`` share qubit 0, and ``CNOT(0,1)`` and
``Rz(1)`` share qubit 1. Likewise ``CNOT(0,1)`` followed by ``CNOT(1,2)``
fuse into a single 3-qubit gate over qubits ``{0,1,2}`` — the two gates
touch different (but overlapping) qubit pairs, which same-qubit-pair
fusion could never merge.

Same-qubit runs (single-qubit gate chains) and same-pair runs (repeated
two-qubit gates on one pair) are just the special cases of this where the
block's qubit set happens to stay fixed at size 1 or 2 — overlapping-
support fusion is a strict generalization of both.

Usage
-----
The simplest way to use fusion is via the ``fuse`` callable attached to
``MPSSimulator``::

    sim = MPSSimulator(chi=64, fusion=True)          # enable fusion globally
    state = sim.run(circuit)

Or call the ``GateFuser`` directly for more control::

    from mps_xtrap.fusion import GateFuser

    fuser = GateFuser(
        max_qubits=3,    # allow blocks spanning up to 3 qubits
        max_window=None, # None -> no cap on how many gates join one block
    )
    fused_circuit = fuser(circuit)       # GateFuser is callable
    state = sim.run(fused_circuit)

``GateFuser`` is also callable with a plain list of ``GateInstruction``
objects, returning a new list — useful when building custom pipelines.

Performance guidance
--------------------
* ``max_qubits=1`` fuses only chains of single-qubit gates on the same
  qubit — zero accuracy impact, minimal overhead.
* ``max_qubits=2`` additionally merges any run of gates whose combined
  support stays within a pair of qubits (the classic "same pair" case,
  plus mixed single/two-qubit runs on that pair).
* ``max_qubits >= 3`` fuses across genuinely overlapping neighborhoods —
  e.g. a chain of two-qubit gates sweeping across a register, as in a
  Trotterized brick-layer circuit — at the cost of a larger dense gate
  (2^k x 2^k) and a k-site SVD sweep to re-split it into the MPS. A good
  default range is 3-4; going much higher trades a bigger one-off
  contraction for diminishing returns and a larger dense matrix.
* ``max_window=w`` caps how many gates are merged in one block regardless
  of qubit count. ``None`` (default) means unlimited.

Notes
-----
* Fusion never changes the mathematical operation — the fused gate is
  exactly the ordered product of the original gates, each embedded into
  the joint space of the block's qubits (identity on qubits the gate
  doesn't touch).
* Custom matrices (``GateInstruction.matrix is not None``) are handled
  seamlessly alongside named gates.
* Fusion only merges gates that are consecutive in the instruction list.
  A gate that doesn't overlap the current block (or would push it past
  ``max_qubits``/``max_window``) closes the block; instructions are never
  reordered, so correctness for non-commuting gates is preserved.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_operator(op: np.ndarray, op_qubits: List[int], union_qubits: List[int]) -> np.ndarray:
    """
    Embed an operator acting on ``op_qubits`` into the full operator space
    spanned by ``union_qubits``, acting as identity on qubits that are in
    ``union_qubits`` but not in ``op_qubits``.

    Parameters
    ----------
    op : ndarray, shape (2**m, 2**m)
        Operator matrix, with leg order matching ``op_qubits``.
    op_qubits : list of int, length m
        The physical qubits ``op`` acts on, in the order matching its legs.
    union_qubits : list of int, length k
        The full (sorted, ascending) qubit set of the fusion block. Must be
        a superset of ``op_qubits``. Defines the canonical leg order of the
        returned operator.

    Returns
    -------
    ndarray, shape (2**k, 2**k)
        The embedded operator over ``union_qubits``, in ascending-qubit
        leg order.
    """
    k = len(union_qubits)
    m = len(op_qubits)
    slot = {q: i for i, q in enumerate(union_qubits)}
    gate_slots = [slot[q] for q in op_qubits]
    other_qubits = [q for q in union_qubits if q not in op_qubits]
    other_slots = [slot[q] for q in other_qubits]

    op_t = np.asarray(op).reshape((2,) * m + (2,) * m)

    if k == m:
        full_t = op_t
        out_src = list(range(m))
        in_src = list(range(m, 2 * m))
        order = gate_slots
    else:
        r = k - m
        ident = np.eye(2 ** r, dtype=complex).reshape((2,) * r + (2,) * r)
        full_t = np.tensordot(op_t, ident, axes=0)
        # Leg layout after tensordot: [op_out(m), op_in(m), id_out(r), id_in(r)]
        out_src = list(range(0, m)) + list(range(2 * m, 2 * m + r))
        in_src = list(range(m, 2 * m)) + list(range(2 * m + r, 2 * m + 2 * r))
        order = gate_slots + other_slots

    perm = [0] * (2 * k)
    for i, s in enumerate(order):
        perm[s] = out_src[i]
        perm[k + s] = in_src[i]
    full_t = full_t.transpose(perm)
    return full_t.reshape(2 ** k, 2 ** k)


def _fuse_block_matrix(block: list, union_qubits: List[int]) -> np.ndarray:
    """
    Compute the combined (2**k, 2**k) matrix for a block of instructions,
    each embedded into ``union_qubits`` and multiplied in circuit order
    (first-applied gate rightmost, matching standard matrix convention).
    """
    combined = None
    for inst in block:
        op = np.asarray(inst.get_matrix())
        m = len(inst.qubits)
        op2d = op.reshape(2 ** m, 2 ** m)
        embedded = _embed_operator(op2d, list(inst.qubits), union_qubits)
        combined = embedded if combined is None else embedded @ combined
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GateFuser:
    """
    Callable that fuses gate sequences in a circuit or instruction list
    based on overlapping (shared-qubit) support.

    Parameters
    ----------
    max_qubits : int
        Maximum number of distinct qubits a fused block may span. A run of
        consecutive gates is merged as long as each new gate shares at
        least one qubit with the block built so far, and the block's total
        qubit count would not exceed ``max_qubits``. Default 3.
    max_window : int or None
        Maximum number of gates to merge in a single fusion block,
        regardless of qubit count. ``None`` (default) means unlimited.

    Examples
    --------
    >>> from mps_xtrap import Circuit
    >>> from mps_xtrap.fusion import GateFuser
    >>> c = Circuit(3)
    >>> c.h(0).cx(0, 1).cx(1, 2)
    >>> fuser = GateFuser(max_qubits=3)
    >>> fc = fuser(c)          # returns a new Circuit
    >>> print(len(c.instructions), '->', len(fc.instructions))
    3 -> 1
    """

    def __init__(
        self,
        max_qubits: int = 3,
        max_window: Optional[int] = None,
    ):
        if max_qubits < 1:
            raise ValueError("max_qubits must be >= 1.")
        self.max_qubits = max_qubits
        self.max_window = max_window

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        circuit_or_instructions,
    ):
        """
        Fuse gates in *circuit_or_instructions*.

        Parameters
        ----------
        circuit_or_instructions : Circuit or list of GateInstruction
            Input to fuse.

        Returns
        -------
        Circuit or list of GateInstruction
            Same type as input, with overlapping-support gate runs merged.
        """
        # Import here to avoid circular imports at module load time
        from .circuit import Circuit
        from .instruction import GateInstruction  # noqa: F401

        if isinstance(circuit_or_instructions, Circuit):
            instructions = circuit_or_instructions.instructions
            fused = self._fuse_instructions(instructions)
            out = Circuit(circuit_or_instructions.n)
            out.instructions = fused
            n_before = len(instructions)
            n_after = len(fused)
            if n_before != n_after:
                logger.debug(
                    "GateFuser: %d instructions -> %d (saved %d gate applications)",
                    n_before, n_after, n_before - n_after,
                )
            return out
        else:
            # Plain list of GateInstruction
            return self._fuse_instructions(list(circuit_or_instructions))

    # ------------------------------------------------------------------
    # Core fusion logic
    # ------------------------------------------------------------------

    def _fuse_instructions(self, instructions: list) -> list:
        """
        Sweep through the instruction list left to right, greedily growing
        a "block" of consecutive gates whose union of qubits overlaps at
        every step and never exceeds ``max_qubits`` (or ``max_window``
        gates). Each maximal block is fused into a single gate.
        """
        if not instructions:
            return []

        output: list = []
        block: list = []
        block_qubits: set = set()

        for inst in instructions:
            inst_qubits = set(inst.qubits)

            if not block:
                block = [inst]
                block_qubits = inst_qubits
                continue

            shares_support = bool(inst_qubits & block_qubits)
            proposed_qubits = block_qubits | inst_qubits
            fits_qubits = len(proposed_qubits) <= self.max_qubits
            fits_window = self.max_window is None or len(block) < self.max_window

            if shares_support and fits_qubits and fits_window:
                block.append(inst)
                block_qubits = proposed_qubits
            else:
                output.extend(self._fuse_block(block, block_qubits))
                block = [inst]
                block_qubits = inst_qubits

        output.extend(self._fuse_block(block, block_qubits))
        return output

    def _fuse_block(self, block: list, block_qubits: set) -> list:
        """
        Fuse a block of overlapping-support instructions into one gate.
        Returns a single-element list, or the block unchanged if it only
        has one instruction.
        """
        if len(block) <= 1:
            return block

        from .instruction import GateInstruction

        union_qubits = sorted(block_qubits)
        k = len(union_qubits)

        try:
            combined = _fuse_block_matrix(block, union_qubits)
        except Exception as exc:
            # Defensive: if fusion fails for any reason, return original block
            logger.warning("GateFuser: block fusion failed (%s), skipping.", exc)
            return block

        names = "+".join(inst.name for inst in block)
        fused_inst = GateInstruction(
            name=f"fused_{k}q",
            qubits=union_qubits,
            params=[],
            matrix=combined.reshape((2,) * k + (2,) * k),
            label=f"[{names}]",
        )
        return [fused_inst]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def fusion_report(self, circuit) -> dict:
        """
        Return a dict summarising what fusion would do (without modifying
        the circuit).

        Returns
        -------
        dict with keys:
            original_count : int
            fused_count : int
            saved : int
            blocks_by_size : dict
                Mapping of fused-block qubit count -> number of such blocks
                (e.g. ``{2: 3, 3: 1}`` means three 2-qubit fused blocks and
                one 3-qubit fused block).
        """
        before = len(circuit.instructions)
        fused_circuit = self(circuit)
        after = len(fused_circuit.instructions)

        blocks_by_size: dict = {}
        for inst in fused_circuit.instructions:
            if inst.name.startswith("fused_") and inst.name.endswith("q"):
                k = len(inst.qubits)
                blocks_by_size[k] = blocks_by_size.get(k, 0) + 1

        return {
            "original_count": before,
            "fused_count": after,
            "saved": before - after,
            "blocks_by_size": blocks_by_size,
        }

    def __repr__(self) -> str:
        return f"GateFuser(max_qubits={self.max_qubits}, max_window={self.max_window})"


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

#: Default fuser: overlapping-support fusion up to 3-qubit blocks.
default_fuser = GateFuser(max_qubits=3)


def fuse(circuit, max_qubits: int = 3, max_window: Optional[int] = None):
    """
    Convenience function: fuse gates in *circuit* and return a new circuit.

    Parameters
    ----------
    circuit : Circuit
        Input circuit.
    max_qubits : int
        Maximum number of qubits a fused block may span (default 3).
    max_window : int or None
        Maximum block size in number of gates (None = unlimited).

    Returns
    -------
    Circuit
        New circuit with fused gate instructions.

    Examples
    --------
    >>> from mps_xtrap import Circuit, MPSSimulator
    >>> from mps_xtrap.fusion import fuse
    >>> c = Circuit(4)
    >>> c.h(0).cx(0, 1).cx(1, 2)   # overlapping support -> fuses to 1 gate
    >>> fc = fuse(c)
    >>> sim = MPSSimulator(chi=64)
    >>> state = sim.run(fc)
    """
    return GateFuser(max_qubits=max_qubits, max_window=max_window)(circuit)


__all__ = ["GateFuser", "fuse", "default_fuser"]
