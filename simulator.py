"""
Applies a :class:`~mps_xtrap.circuits.circuit.Circuit` to an
:class:`~mps_xtrap.core.mps.MPS`, performing SVD truncations at each
multi-qubit gate — single SVDs for two-qubit gates, and a generalized
multi-site SVD sweep for the wider gates produced by overlapping-support
gate fusion (see :mod:`mps_xtrap.fusion`).
"""

from __future__ import annotations
import logging
from typing import Optional, Sequence

import numpy as np

from .mps import MPS
from .circuit import Circuit
from .instruction import GateInstruction

logger = logging.getLogger(__name__)


class MPSSimulator:
    """
    Applies a Circuit to an MPS state.

    Two-qubit gates on non-adjacent qubits are handled by first
    swapping qubits into adjacent positions, applying the gate,
    then swapping back.

    Parameters
    ----------
    chi : int
        Maximum bond dimension for SVD truncation.
    svd_threshold : float
        Singular values below this are discarded before bond-dim cap.
    device : str
        'cpu' (default) or 'cuda'. With 'cuda', all tensor operations
        run on the GPU via CuPy. Falls back to CPU automatically if
        CuPy is not installed or no GPU is detected.
    fusion : bool or GateFuser
        Gate fusion strategy applied before simulation.

        * ``False`` (default) — no fusion.
        * ``True`` — use the default :class:`~mps_xtrap.fusion.GateFuser`,
          which fuses consecutive gates by *overlapping (shared-qubit)
          support* into blocks of up to 3 qubits.
        * A :class:`~mps_xtrap.fusion.GateFuser` instance — use that fuser
          directly, giving full control over ``max_qubits`` and
          ``max_window``.

        Fusion never changes simulation accuracy; it only reduces the number
        of gate applications (and therefore SVD calls) by merging runs of
        gates that share qubit support into a single combined gate. Fused
        blocks spanning 3 or more qubits are applied via a generalized
        multi-site SVD sweep (see ``_apply_multi``), after gathering the
        block's qubits into adjacent positions with a SWAP network if
        needed.

    Examples
    --------
    >>> sim = MPSSimulator(chi=64, fusion=True)           # default fusion
    >>> state = sim.run(circuit)

    >>> from mps_xtrap.fusion import GateFuser
    >>> fuser = GateFuser(max_qubits=4, max_window=None)
    >>> sim = MPSSimulator(chi=64, fusion=fuser)
    >>> state = sim.run(circuit)
    """

    def __init__(
        self,
        chi: int,
        svd_threshold: float = 1e-14,
        device: str = 'cpu',
        fusion=False,
    ):
        self.chi = chi
        self.svd_threshold = svd_threshold
        self.device = device

        # Resolve fusion argument to None or a GateFuser instance
        if fusion is False or fusion is None:
            self._fuser = None
        elif fusion is True:
            from .fusion import GateFuser
            self._fuser = GateFuser(max_qubits=3)
        else:
            # Assume a GateFuser (or any callable that accepts a Circuit)
            self._fuser = fusion

    # ------------------------------------------------------------------
    # Public property for inspection / replacement after construction
    # ------------------------------------------------------------------

    @property
    def fusion(self):
        """The active :class:`~mps_xtrap.fusion.GateFuser`, or ``None``."""
        return self._fuser

    @fusion.setter
    def fusion(self, value):
        from .fusion import GateFuser
        if value is False or value is None:
            self._fuser = None
        elif value is True:
            self._fuser = GateFuser(max_qubits=3)
        else:
            self._fuser = value

    def run(self, circuit: Circuit) -> MPS:
        """
        Simulate circuit from |0...0> and return final MPS state.

        If a fusion strategy is configured (``fusion=True`` or a
        :class:`~mps_xtrap.fusion.GateFuser` instance was passed at
        construction), the circuit is fused before simulation.  The
        original circuit object is never modified.

        Returns
        -------
        MPS
            Final state after all gates applied (on self.device).
        """
        if self._fuser is not None:
            circuit = self._fuser(circuit)
        state = MPS(circuit.n, self.chi, device=self.device)
        self._apply_circuit(circuit, state)
        return state

    def run_from(self, circuit: Circuit, state: MPS) -> MPS:
        """Apply circuit to an existing MPS state (modifies a copy).

        Fusion (if configured) is applied before simulation.
        """
        if self._fuser is not None:
            circuit = self._fuser(circuit)
        new_state = state.copy()
        new_state.chi = self.chi
        self._apply_circuit(circuit, new_state)
        return new_state

    def _apply_circuit(self, circuit: Circuit, state: MPS):
        for inst in circuit.instructions:
            n_qubits = len(inst.qubits)
            if n_qubits == 1:
                self._apply_single(inst, state)
            elif n_qubits == 2:
                self._apply_two(inst, state)
            elif n_qubits >= 3:
                self._apply_multi(inst, state)
            else:
                raise ValueError(f"Gate instruction has no target qubits: {inst}")

    def _apply_single(self, inst: GateInstruction, state: MPS):
        """Apply single-qubit gate via matrix multiplication on physical index."""
        from .mps import _to_backend
        xp = state.xp
        q = inst.qubits[0]
        gate = _to_backend(inst.get_matrix(), xp)   # (2,2) on device
        t = state.get_tensor(q)                      # (chi_l, 2, chi_r) on device
        state.set_tensor(q, xp.einsum("sp,lpr->lsr", gate, t))
        state.center = None

    def _apply_two(self, inst: GateInstruction, state: MPS):
        """Apply two-qubit gate with SVD truncation."""
        from .mps import _to_backend
        xp = state.xp
        q1, q2 = inst.qubits

        # Handle non-adjacent qubits via SWAP chain
        if abs(q1 - q2) != 1:
            self._apply_two_nonlocal(inst, state)
            return

        # Ensure q1 < q2
        if q1 > q2:
            q1, q2 = q2, q1
            gate = inst.get_matrix()
            gate = gate.transpose(1, 0, 3, 2)   # swap qubit ordering
        else:
            gate = inst.get_matrix()

        gate = _to_backend(gate, xp)

        # Build theta: two-site tensor
        A = state.get_tensor(q1)   # (chi_l, 2, chi_m)
        B = state.get_tensor(q2)   # (chi_m, 2, chi_r)
        theta = xp.einsum("lir,rjs->lijs", A, B)            # (chi_l, 2, 2, chi_r)
        theta = xp.einsum("abij,lijs->labs", gate, theta)   # apply gate

        state.apply_svd_truncation(q1, theta, chi=self.chi, svd_threshold=self.svd_threshold)
        state.center = None

    def _apply_two_nonlocal(self, inst: GateInstruction, state: MPS):
        """
        Apply a gate on non-adjacent qubits via SWAP chain.
        Brings qubits adjacent, applies gate, swaps back.
        """
        from .gates import SWAP as make_swap
        q1, q2 = inst.qubits
        if q1 > q2:
            q1, q2 = q2, q1

        # Bubble q2 adjacent to q1
        swap_gate = make_swap()
        swap_inst = GateInstruction('SWAP', [0, 1], matrix=swap_gate)

        for pos in range(q2, q1 + 1, -1):
            swap_inst.qubits = [pos - 1, pos]
            swap_inst.matrix = swap_gate
            self._apply_two(swap_inst, state)

        # Now apply actual gate at (q1, q1+1)
        modified_inst = GateInstruction(
            inst.name, [q1, q1 + 1],
            inst.params, inst.matrix, inst.label
        )
        self._apply_two(modified_inst, state)

        # Swap back
        for pos in range(q1 + 1, q2):
            swap_inst.qubits = [pos, pos + 1]
            self._apply_two(swap_inst, state)

    def _swap_adjacent(self, state: MPS, pos: int):
        """Apply a SWAP gate between adjacent positions (pos, pos + 1)."""
        from .gates import SWAP as make_swap
        self._apply_two(GateInstruction('SWAP', [pos, pos + 1], matrix=make_swap()), state)

    def _compute_gather_swaps(self, sorted_qubits: list, n: int) -> list:
        """
        Compute the sequence of adjacent-position swaps that gathers
        ``sorted_qubits`` (ascending, distinct) into contiguous positions
        ``[sorted_qubits[0], ..., sorted_qubits[0] + k - 1]``, without ever
        moving ``sorted_qubits[0]`` itself.

        Simulates the swap network on a permutation array (no MPS tensors
        touched here) purely to work out which adjacent positions to swap,
        and in what order. Applying the returned list of positions in order
        performs the gather; applying it in *reverse* order afterwards
        exactly restores the original qubit-to-site layout (every adjacent
        SWAP is its own inverse).

        Returns
        -------
        list of int
            Positions ``a`` such that each step swaps sites ``(a, a + 1)``.
        """
        perm = list(range(n))       # perm[pos] = qubit currently at pos
        loc = {q: q for q in range(n)}  # loc[qubit] = current pos
        swaps = []
        p0 = sorted_qubits[0]

        for i, q in enumerate(sorted_qubits[1:], start=1):
            target = p0 + i
            cur = loc[q]
            while cur > target:
                left_q, right_q = perm[cur - 1], perm[cur]
                perm[cur - 1], perm[cur] = right_q, left_q
                loc[right_q] = cur - 1
                loc[left_q] = cur
                swaps.append(cur - 1)
                cur -= 1

        return swaps

    def _apply_multi(self, inst: GateInstruction, state: MPS):
        """
        Apply a fused gate spanning 3 or more qubits (as produced by
        overlapping-support gate fusion).

        The qubits need not be adjacent or given in sorted order: they are
        first gathered into contiguous positions via a SWAP network (built
        by ``_compute_gather_swaps``), the block is contracted with the
        gate and split back apart with a generalized multi-site SVD sweep
        (``MPS.apply_block_svd``), and finally the same swaps are undone in
        reverse to restore the original qubit-to-site layout — the same
        "swap in / apply / swap out" strategy ``_apply_two_nonlocal`` uses
        for a single non-adjacent pair, generalized to a whole block.
        """
        from .mps import _to_backend
        xp = state.xp

        qubits = list(inst.qubits)
        k = len(qubits)
        sorted_qubits = sorted(qubits)

        gate_t = np.asarray(inst.get_matrix()).reshape((2,) * k + (2,) * k)
        if qubits != sorted_qubits:
            # Permute gate legs to match ascending qubit order.
            order = [qubits.index(q) for q in sorted_qubits]
            gate_t = gate_t.transpose(order + [k + o for o in order])

        swaps = self._compute_gather_swaps(sorted_qubits, state.n)
        for pos in swaps:
            self._swap_adjacent(state, pos)

        start = sorted_qubits[0]
        gate = _to_backend(gate_t, xp)

        # Contract the k adjacent site tensors into one block tensor:
        # (chi_l, 2, 2, ..., 2 [k times], chi_r)
        big = state.get_tensor(start)
        for i in range(1, k):
            t = state.get_tensor(start + i)
            big = xp.tensordot(big, t, axes=([big.ndim - 1], [0]))

        # Contract the gate's "in" legs with the block's physical legs.
        phys_axes = list(range(1, k + 1))
        gate_in_axes = list(range(k, 2 * k))
        new = xp.tensordot(gate, big, axes=(gate_in_axes, phys_axes))
        # `new` layout: [gate_out (k legs), chi_l, chi_r] -> reorder to
        # (chi_l, 2, ..., 2 [k], chi_r)
        perm = [k] + list(range(k)) + [k + 1]
        block_tensor = new.transpose(perm)

        state.apply_block_svd(start, block_tensor, k, chi=self.chi, svd_threshold=self.svd_threshold)
        state.center = None

        for pos in reversed(swaps):
            self._swap_adjacent(state, pos)

    def expectation_value(
        self,
        state: MPS,
        observable: str,
        site: int,
        site2: Optional[int] = None,
    ) -> float:
        """
        Convenience wrapper to compute expectation values.

        Parameters
        ----------
        state : MPS
        observable : str
            'X', 'Y', 'Z', 'ZZ', 'XX', etc.
        site : int
            Primary site.
        site2 : int, optional
            Second site for two-qubit observables.
        """
        if site2 is None:
            if observable == 'Z':
                return state.expectation_pauli_z(site)
            elif observable == 'X':
                return state.expectation_pauli_x(site)
            elif observable == 'Y':
                return state.expectation_pauli_y(site)
            else:
                raise ValueError(f"Unknown single-site observable: {observable}")

        # Two-site correlator, e.g. 'ZZ', 'XX', 'YY', 'XZ', 'ZY', ...
        if site == site2:
            raise ValueError("site and site2 must refer to different qubits.")
        if len(observable) != 2 or any(c not in 'XYZ' for c in observable):
            raise ValueError(
                f"Unknown two-site observable: {observable!r}. "
                "Use a two-character Pauli string such as 'ZZ', 'XX', 'XZ', etc."
            )

        from .gates import X, Y, Z
        paulis = {'X': X, 'Y': Y, 'Z': Z}
        op_i = paulis[observable[0]]()
        op_j = paulis[observable[1]]()

        val = state.expectation_two_site(op_i, op_j, site, site2)
        return float(np.real(val))

    def expectation_correlator(
        self,
        state: MPS,
        observable: str,
        sites: Sequence[int],
    ) -> float:
        """
        Compute an N-site Pauli-string correlator for any number of sites,
        adjacent or not, e.g. <Z0 X2 Y5>.

        Parameters
        ----------
        state : MPS
        observable : str
            A Pauli string, one character ('X', 'Y', or 'Z') per site,
            e.g. 'ZZ', 'XYZ', 'ZZZZ'. Length must match len(sites).
        sites : sequence of int
            Site indices, one per character of `observable`. Must all be
            distinct, but need not be adjacent or given in sorted order.

        Returns
        -------
        float
            The real part of <psi| P_0(sites[0]) ... P_k(sites[k]) |psi>.

        Notes
        -----
        For a single site, use expectation_value instead. For exactly two
        sites, this is equivalent to expectation_value(state, observable,
        sites[0], sites[1]).
        """
        if len(observable) != len(sites):
            raise ValueError(
                f"observable {observable!r} has {len(observable)} characters "
                f"but {len(sites)} sites were given; these must match."
            )
        if len(sites) == 0:
            raise ValueError("Need at least one site.")
        if any(c not in 'XYZ' for c in observable):
            raise ValueError(
                f"Unknown observable {observable!r}; use only 'X', 'Y', 'Z' characters."
            )
        if len(set(sites)) != len(sites):
            raise ValueError("All sites must be distinct.")

        if len(sites) == 1:
            return self.expectation_value(state, observable, sites[0])

        from .gates import X, Y, Z
        paulis = {'X': X, 'Y': Y, 'Z': Z}
        ops = [paulis[c]() for c in observable]

        val = state.expectation_multi_site(ops, list(sites))
        return float(np.real(val))
