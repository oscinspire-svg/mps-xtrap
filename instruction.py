"""
Gate instruction representation for :mod:`mps_xtrap.circuits`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from numpy.typing import NDArray

from .gates import get_gate


@dataclass
class GateInstruction:
    """
    A single gate application.

    Attributes
    ----------
    name : str
        Gate name (e.g., 'H', 'CNOT', 'Rx').
    qubits : list of int
        Target qubits. Length 1 for single-qubit gates, 2 for two-qubit.
    params : list of float
        Gate parameters (e.g., rotation angles).
    matrix : ndarray or None
        Explicit gate matrix (overrides name lookup if provided).
    label : str
        Human-readable label for circuit diagrams.
    """
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    matrix: Optional[NDArray] = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name
        if not isinstance(self.qubits, list):
            self.qubits = list(self.qubits)

    def get_matrix(self) -> NDArray:
        if self.matrix is not None:
            return self.matrix
        return get_gate(self.name, *self.params)

    def __repr__(self) -> str:
        q = ",".join(map(str, self.qubits))
        p = f"({','.join(f'{x:.4f}' for x in self.params)})" if self.params else ""
        return f"{self.name}{p}[{q}]"
