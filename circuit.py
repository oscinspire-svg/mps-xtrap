"""
Quantum circuit representation: an ordered list of gate instructions with
a fluent builder API.
"""

from __future__ import annotations
from typing import List

from numpy.typing import NDArray

from .instruction import GateInstruction


class Circuit:
    """
    A quantum circuit as an ordered list of gate instructions.

    Supports a fluent builder API:
        c = Circuit(4)
        c.h(0).cx(0,1).rz(np.pi/4, 2).measure_all()
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.instructions: List[GateInstruction] = []

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def _add(self, name: str, qubits, params=None, matrix=None) -> 'Circuit':
        if isinstance(qubits, int):
            qubits = [qubits]
        inst = GateInstruction(
            name=name,
            qubits=list(qubits),
            params=list(params) if params else [],
            matrix=matrix,
        )
        self.instructions.append(inst)
        return self

    # Single-qubit gates
    def i(self, q): return self._add('I', [q])
    def x(self, q): return self._add('X', [q])
    def y(self, q): return self._add('Y', [q])
    def z(self, q): return self._add('Z', [q])
    def h(self, q): return self._add('H', [q])
    def s(self, q): return self._add('S', [q])
    def t(self, q): return self._add('T', [q])
    def sdg(self, q): return self._add('Sdg', [q])
    def tdg(self, q): return self._add('Tdg', [q])

    def rx(self, theta: float, q: int): return self._add('Rx', [q], [theta])
    def ry(self, theta: float, q: int): return self._add('Ry', [q], [theta])
    def rz(self, theta: float, q: int): return self._add('Rz', [q], [theta])
    def p(self, phi: float, q: int): return self._add('P', [q], [phi])
    def u(self, theta, phi, lam, q): return self._add('U', [q], [theta, phi, lam])

    # Two-qubit gates
    def cx(self, ctrl, tgt): return self._add('CNOT', [ctrl, tgt])
    def cnot(self, ctrl, tgt): return self._add('CNOT', [ctrl, tgt])
    def cz(self, q1, q2): return self._add('CZ', [q1, q2])
    def swap(self, q1, q2): return self._add('SWAP', [q1, q2])
    def iswap(self, q1, q2): return self._add('iSWAP', [q1, q2])
    def xx(self, theta, q1, q2): return self._add('XX', [q1, q2], [theta])
    def yy(self, theta, q1, q2): return self._add('YY', [q1, q2], [theta])
    def zz(self, theta, q1, q2): return self._add('ZZ', [q1, q2], [theta])
    def crz(self, theta, ctrl, tgt): return self._add('CRz', [ctrl, tgt], [theta])
    def cp(self, phi, ctrl, tgt): return self._add('CP', [ctrl, tgt], [phi])

    # Custom matrix
    def unitary(self, matrix: NDArray, qubits): return self._add('U_custom', qubits, matrix=matrix)

    # ------------------------------------------------------------------
    # Helpers for common circuit patterns
    # ------------------------------------------------------------------

    def barrier(self, *qubits):
        """Barrier is a no-op for simulation but marks a layer boundary."""
        return self  # no-op

    def __len__(self):
        return len(self.instructions)

    def depth(self) -> int:
        """Estimate circuit depth (number of non-overlapping gate layers)."""
        last_use = [-1] * self.n
        max_layer = -1
        for inst in self.instructions:
            d = max(last_use[q] for q in inst.qubits) + 1
            max_layer = max(max_layer, d)
            for q in inst.qubits:
                last_use[q] = d
        return max_layer + 1  # convert 0-indexed layer to count

    def two_qubit_count(self) -> int:
        return sum(1 for inst in self.instructions if len(inst.qubits) == 2)

    def __repr__(self) -> str:
        return (f"Circuit(n={self.n}, gates={len(self.instructions)}, "
                f"depth≈{self.depth()}, 2q={self.two_qubit_count()})")

    def draw(self) -> str:
        """Simple text-based circuit diagram."""
        lines = [f"q{i}: " for i in range(self.n)]
        for inst in self.instructions:
            label = str(inst)
            for i, q in enumerate(inst.qubits):
                lines[q] += f"──{label}──"
            # Pad other qubits
            max_len = max(len(l) for l in lines)
            lines = [l + '─' * (max_len - len(l)) for l in lines]
        return '\n'.join(lines)
