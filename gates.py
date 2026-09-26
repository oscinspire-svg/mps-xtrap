"""
Quantum gate definitions.

All gates are numpy arrays. Single-qubit gates have shape (2, 2).
Two-qubit gates have shape (2, 2, 2, 2) in the convention op[s1', s2', s1, s2].
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Pauli and Hadamard
# ---------------------------------------------------------------------------

def I() -> NDArray:
    return np.eye(2, dtype=complex)

def X() -> NDArray:
    return np.array([[0, 1], [1, 0]], dtype=complex)

def Y() -> NDArray:
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def Z() -> NDArray:
    return np.array([[1, 0], [0, -1]], dtype=complex)

def H() -> NDArray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

def S() -> NDArray:
    return np.array([[1, 0], [0, 1j]], dtype=complex)

def T() -> NDArray:
    return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)

def Sdg() -> NDArray:
    return np.array([[1, 0], [0, -1j]], dtype=complex)

def Tdg() -> NDArray:
    return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)


# ---------------------------------------------------------------------------
# Rotation gates
# ---------------------------------------------------------------------------

def Rx(theta: float) -> NDArray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def Ry(theta: float) -> NDArray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz(theta: float) -> NDArray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def P(phi: float) -> NDArray:
    """Phase gate: |1> → e^{iφ} |1>."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

def U(theta: float, phi: float, lam: float) -> NDArray:
    """General single-qubit unitary U3(θ, φ, λ)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Two-qubit gates (shape: (2,2,2,2) with convention out[s1',s2',s1,s2])
# ---------------------------------------------------------------------------

def _two_qubit_from_matrix(mat: NDArray) -> NDArray:
    """Reshape a 4x4 matrix into (2,2,2,2) tensor."""
    return mat.reshape(2, 2, 2, 2).transpose(0, 1, 2, 3)


def CNOT() -> NDArray:
    mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    return mat.reshape(2, 2, 2, 2)


def CZ() -> NDArray:
    mat = np.diag([1, 1, 1, -1]).astype(complex)
    return mat.reshape(2, 2, 2, 2)


def SWAP() -> NDArray:
    mat = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)
    return mat.reshape(2, 2, 2, 2)


def iSWAP() -> NDArray:
    mat = np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)
    return mat.reshape(2, 2, 2, 2)


def XX(theta: float) -> NDArray:
    """Ising XX interaction: exp(-i θ XX / 2)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    mat = np.array([
        [c, 0, 0, -1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [-1j * s, 0, 0, c],
    ], dtype=complex)
    return mat.reshape(2, 2, 2, 2)


def YY(theta: float) -> NDArray:
    """Ising YY interaction: exp(-i θ YY / 2)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    mat = np.array([
        [c, 0, 0, 1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [1j * s, 0, 0, c],
    ], dtype=complex)
    return mat.reshape(2, 2, 2, 2)


def ZZ(theta: float) -> NDArray:
    """Ising ZZ interaction: exp(-i θ ZZ / 2)."""
    return np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2),
    ]).reshape(2, 2, 2, 2)


def CRz(theta: float) -> NDArray:
    """Controlled-Rz."""
    mat = np.diag([1, 1, np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)
    return mat.reshape(2, 2, 2, 2)


def CP(phi: float) -> NDArray:
    """Controlled phase."""
    mat = np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)
    return mat.reshape(2, 2, 2, 2)


# ---------------------------------------------------------------------------
# Gate registry for string-based lookup
# ---------------------------------------------------------------------------

SINGLE_QUBIT_GATES = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z,
    'H': H,
    'S': S,
    'T': T,
    'Sdg': Sdg,
    'Tdg': Tdg,
    'Rx': Rx,
    'Ry': Ry,
    'Rz': Rz,
    'P': P,
    'U': U,
}

TWO_QUBIT_GATES = {
    'CNOT': CNOT,
    'CX': CNOT,
    'CZ': CZ,
    'SWAP': SWAP,
    'iSWAP': iSWAP,
    'XX': XX,
    'YY': YY,
    'ZZ': ZZ,
    'CRz': CRz,
    'CP': CP,
}


def get_gate(name: str, *params) -> NDArray:
    """
    Retrieve a gate by name, optionally with parameters.

    Examples
    --------
    get_gate('H')
    get_gate('Rx', np.pi / 2)
    get_gate('CNOT')
    """
    name = name.strip()
    if name in SINGLE_QUBIT_GATES:
        fn = SINGLE_QUBIT_GATES[name]
        return fn(*params) if params else fn()
    if name in TWO_QUBIT_GATES:
        fn = TWO_QUBIT_GATES[name]
        return fn(*params) if params else fn()
    raise ValueError(f"Unknown gate: '{name}'. "
                     f"Known gates: {sorted(SINGLE_QUBIT_GATES) + sorted(TWO_QUBIT_GATES)}")
