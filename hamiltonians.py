"""
Predefined Hamiltonian-simulation circuit builders.

This module ships Trotterized time-evolution circuit *builders* for four
standard models used throughout condensed-matter and quantum-chemistry
benchmarking:

  * Transverse-Field Ising Model  -> :func:`tfim_circuit`
  * Heisenberg model (XXX/XXZ)    -> :func:`heisenberg_circuit`
  * Hydrogen molecule (H2)        -> :func:`h2_circuit`
  * Fermi-Hubbard model           -> :func:`fermi_hubbard_circuit`

Nothing about system size, step size, total time, couplings, boundary
conditions, or the observable/site to measure is hardcoded — every one of
those is a keyword argument.

Design notes
------------
* Every builder's *first* positional argument is ``dt`` and every other
  physical parameter is a keyword argument with **no default** for the
  parameters that define the problem instance (``n_qubits``/``n_sites``,
  ``t_total``). This keeps the ``circuit_fn(dt)`` calling convention that
  :class:`~mps_xtrap.extrapolation.TrotterExtrapolator` and
  :class:`~mps_xtrap.extrapolation.TrotterSweepConfig` use, once the other
  keywords are bound — see :func:`bind_circuit`, which does that binding
  for you so a Trotter dt-sweep "just works" without hand-written
  ``functools.partial`` calls.
* :func:`simulate_hamiltonian` runs a single circuit at one ``dt`` and
  returns either plain numbers or a :class:`HamiltonianResult` (with
  metadata), your choice, via ``result_mode``. For a dt-sweep with
  Richardson extrapolation, drive
  :class:`~mps_xtrap.extrapolation.TrotterExtrapolator` yourself with a
  circuit function from :func:`bind_circuit` — see the README.

Physics conventions
--------------------
TFIM:        H = -J * sum_<i,j> Z_i Z_j  -  h * sum_i X_i
Heisenberg:  H = sum_<i,j> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) - hz * sum_i Z_i
H2:          H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*Y0Y1 + g5*X0X1
             (2-qubit Bravyi-Kitaev-tapered qubit Hamiltonian; default
             coefficients are the R=0.75 A values from O'Malley et al.,
             Phys. Rev. X 6, 031007 (2016), Table I — pass your own
             g0..g5 for any other bond length/basis set.)
Fermi-Hubbard: H = -t * sum_{i,sigma} (c^dag_{i,sigma} c_{i+1,sigma} + h.c.)
               + U * sum_i n_{i,up} n_{i,down} - mu * sum_{i,sigma} n_{i,sigma}
               (Jordan-Wigner mapped, qubits ordered spin-block-major:
               qubit = spin * n_sites + site, so intra-spin hopping is a
               nearest-neighbor gate and the on-site U term is the only
               long-range gate — handled transparently by MPSSimulator's
               non-adjacent-qubit swap network.)

Note on the default initial state
----------------------------------
Every circuit here starts from ``|0...0>``. For several of these models
that's a symmetry-protected fixed point of the dynamics — e.g. the
isotropic XX+YY terms in the Heisenberg model conserve total
magnetization, and H2's default X0X1/Y0Y1 coefficients happen to be
equal, which cancels their effect on ``|00>`` — so you'll see no dynamics
at all in those cases. That's genuinely correct physics, not a simulator
quirk. TFIM is the exception: its transverse field does not leave
``|0...0>`` invariant.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .circuit import Circuit
from .simulator import MPSSimulator

__all__ = [
    # Circuit builders
    "tfim_circuit",
    "heisenberg_circuit",
    "h2_circuit",
    "h2_energy",
    "fermi_hubbard_circuit",
    "MODEL_BUILDERS",
    "bind_circuit",
    # Observable spec & result types
    "ObservableSpec",
    "ResultMode",
    "HamiltonianResult",
    # Runner
    "simulate_hamiltonian",
    # H2 reference data
    "H2_DEFAULT_COEFFICIENTS",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _validate_boundary(boundary: str) -> None:
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary!r}")


def _bonds(n: int, boundary: str) -> List[Tuple[int, int]]:
    """Nearest-neighbor bonds for a 1D chain of n sites."""
    bonds = [(i, i + 1) for i in range(n - 1)]
    if boundary == "periodic" and n > 2:
        bonds.append((n - 1, 0))
    return bonds


def _n_steps(dt: float, t_total: float) -> int:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_total < 0:
        raise ValueError("t_total must be non-negative.")
    return max(1, round(t_total / dt))


# ---------------------------------------------------------------------------
# 1. Transverse-Field Ising Model
# ---------------------------------------------------------------------------

def tfim_circuit(
    dt: float,
    n_qubits: int,
    t_total: float,
    J: float = 1.0,
    h: float = 1.0,
    boundary: str = "open",
    order: int = 2,
) -> Circuit:
    """
    Trotterized time evolution under the transverse-field Ising model

        H = -J * sum_<i,j> Z_i Z_j  -  h * sum_i X_i

    Parameters
    ----------
    dt : float
        Trotter step size.
    n_qubits : int
        Chain length.
    t_total : float
        Total evolution time; number of Trotter steps is
        ``round(t_total / dt)`` (at least 1).
    J : float
        ZZ coupling strength.
    h : float
        Transverse-field strength.
    boundary : {'open', 'periodic'}
    order : {1, 2}
        1 = first-order Lie-Trotter, 2 = second-order (Strang) splitting.
    """
    _validate_boundary(boundary)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    steps = _n_steps(dt, t_total)
    bonds = _bonds(n_qubits, boundary)

    c = Circuit(n_qubits)

    def field_layer(sub_dt: float) -> None:
        for q in range(n_qubits):
            c.rx(-2.0 * h * sub_dt, q)

    def ising_layer(sub_dt: float) -> None:
        for (i, j) in bonds:
            c.zz(-2.0 * J * sub_dt, i, j)

    for _ in range(steps):
        if order == 1:
            field_layer(dt)
            ising_layer(dt)
        else:
            field_layer(dt / 2)
            ising_layer(dt)
            field_layer(dt / 2)

    return c


# ---------------------------------------------------------------------------
# 2. Heisenberg model (XXX / XXZ)
# ---------------------------------------------------------------------------

def heisenberg_circuit(
    dt: float,
    n_qubits: int,
    t_total: float,
    model: str = "XXX",
    J: float = 1.0,
    delta: float = 1.0,
    hz: float = 0.0,
    boundary: str = "open",
    order: int = 2,
) -> Circuit:
    """
    Trotterized time evolution under the (an)isotropic Heisenberg model

        H = sum_<i,j> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) - hz * sum_i Z_i

    Parameters
    ----------
    dt, n_qubits, t_total, boundary, order : see :func:`tfim_circuit`.
    model : {'XXX', 'XXZ'}
        'XXX' (isotropic): Jx = Jy = Jz = J.
        'XXZ' (anisotropic): Jx = Jy = J, Jz = delta * J.
        For full independent control, ignore ``model``/``delta`` and pass
        an already-built Jx, Jy, Jz combination via a thin wrapper.
    J : float
        Overall exchange coupling.
    delta : float
        Anisotropy parameter, used only when ``model='XXZ'``.
    hz : float
        Longitudinal field strength.
    """
    _validate_boundary(boundary)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    model = model.upper()
    if model == "XXX":
        Jx = Jy = Jz = J
    elif model == "XXZ":
        Jx = Jy = J
        Jz = delta * J
    else:
        raise ValueError("model must be 'XXX' or 'XXZ'.")

    steps = _n_steps(dt, t_total)
    bonds = _bonds(n_qubits, boundary)

    c = Circuit(n_qubits)

    def field_layer(sub_dt: float) -> None:
        if hz == 0.0:
            return
        for q in range(n_qubits):
            c.rz(-2.0 * hz * sub_dt, q)

    def bond_layer(sub_dt: float) -> None:
        for (i, j) in bonds:
            if Jx != 0.0:
                c.xx(2.0 * Jx * sub_dt, i, j)
            if Jy != 0.0:
                c.yy(2.0 * Jy * sub_dt, i, j)
            if Jz != 0.0:
                c.zz(2.0 * Jz * sub_dt, i, j)

    for _ in range(steps):
        if order == 1:
            field_layer(dt)
            bond_layer(dt)
        else:
            field_layer(dt / 2)
            bond_layer(dt)
            field_layer(dt / 2)

    return c


# ---------------------------------------------------------------------------
# 3. Hydrogen molecule (H2), 2-qubit Bravyi-Kitaev-tapered Hamiltonian
# ---------------------------------------------------------------------------

# R = 0.75 Angstrom values, Table I of O'Malley et al., "Scalable Quantum
# Simulation of Molecular Energies", Phys. Rev. X 6, 031007 (2016).
# Swap in your own coefficients (e.g. from PySCF/OpenFermion/PennyLane) for
# any other bond length or basis set — these are just a physically real
# default so the function works out of the box.
H2_DEFAULT_COEFFICIENTS: Dict[str, float] = {
    "g0": -0.4804,
    "g1": 0.3435,
    "g2": -0.4347,
    "g3": 0.5716,
    "g4": 0.0910,
    "g5": 0.0910,
    "nuclear_repulsion": 0.7055696146,
    "bond_length_angstrom": 0.75,
}


def h2_circuit(
    dt: float,
    t_total: float,
    g0: Optional[float] = None,
    g1: Optional[float] = None,
    g2: Optional[float] = None,
    g3: Optional[float] = None,
    g4: Optional[float] = None,
    g5: Optional[float] = None,
    order: int = 2,
) -> Circuit:
    """
    Trotterized time evolution under the 2-qubit tapered H2 electronic
    Hamiltonian

        H = g0*I + g1*Z0 + g2*Z1 + g3*Z0*Z1 + g4*Y0*Y1 + g5*X0*X1

    Always exactly 2 qubits (this is the tapered/reduced representation).
    Any coefficient left as ``None`` falls back to the R=0.75 A reference
    value in :data:`H2_DEFAULT_COEFFICIENTS` — pass your own set (e.g.
    computed at a different bond length) to override.

    Notes
    -----
    The g0*I term is a global phase under time evolution and is skipped
    (it never affects any observable). To get the *total* electronic
    energy for a given state — e.g. for a ground-state / VQE-style check
    rather than dynamics — use ``h2_energy(state)``.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    coeffs = dict(H2_DEFAULT_COEFFICIENTS)
    for name, val in (("g0", g0), ("g1", g1), ("g2", g2), ("g3", g3), ("g4", g4), ("g5", g5)):
        if val is not None:
            coeffs[name] = val

    steps = _n_steps(dt, t_total)
    c = Circuit(2)

    g1v, g2v, g3v, g4v, g5v = coeffs["g1"], coeffs["g2"], coeffs["g3"], coeffs["g4"], coeffs["g5"]

    def single_layer(sub_dt: float) -> None:
        c.rz(2.0 * g1v * sub_dt, 0)
        c.rz(2.0 * g2v * sub_dt, 1)

    def two_qubit_layer(sub_dt: float) -> None:
        c.zz(2.0 * g3v * sub_dt, 0, 1)
        c.yy(2.0 * g4v * sub_dt, 0, 1)
        c.xx(2.0 * g5v * sub_dt, 0, 1)

    for _ in range(steps):
        if order == 1:
            single_layer(dt)
            two_qubit_layer(dt)
        else:
            single_layer(dt / 2)
            two_qubit_layer(dt)
            single_layer(dt / 2)

    return c


def h2_energy(
    state,
    g0: Optional[float] = None,
    g1: Optional[float] = None,
    g2: Optional[float] = None,
    g3: Optional[float] = None,
    g4: Optional[float] = None,
    g5: Optional[float] = None,
    include_nuclear_repulsion: bool = True,
) -> float:
    """
    Expectation value of the (untruncated) H2 electronic Hamiltonian in
    ``state``, i.e. <state| H |state> (+ nuclear repulsion, by default) —
    a static energy readout, independent of any time evolution.
    """
    coeffs = dict(H2_DEFAULT_COEFFICIENTS)
    for name, val in (("g0", g0), ("g1", g1), ("g2", g2), ("g3", g3), ("g4", g4), ("g5", g5)):
        if val is not None:
            coeffs[name] = val

    e = coeffs["g0"]
    e += coeffs["g1"] * state.expectation_pauli_z(0)
    e += coeffs["g2"] * state.expectation_pauli_z(1)

    from .gates import X, Y, Z
    e += coeffs["g3"] * float(np.real(state.expectation_two_site(Z(), Z(), 0, 1)))
    e += coeffs["g4"] * float(np.real(state.expectation_two_site(Y(), Y(), 0, 1)))
    e += coeffs["g5"] * float(np.real(state.expectation_two_site(X(), X(), 0, 1)))

    if include_nuclear_repulsion:
        e += coeffs["nuclear_repulsion"]
    return float(e)


# ---------------------------------------------------------------------------
# 4. Fermi-Hubbard model
# ---------------------------------------------------------------------------

def fermi_hubbard_circuit(
    dt: float,
    n_sites: int,
    t_total: float,
    t_hop: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
    boundary: str = "open",
    order: int = 1,
) -> Circuit:
    """
    Trotterized time evolution under the 1D Fermi-Hubbard model

        H = -t_hop * sum_{i,sigma} (c^dag_{i,sigma} c_{i+1,sigma} + h.c.)
            + U * sum_i n_{i,up} n_{i,down}
            - mu * sum_{i,sigma} n_{i,sigma}

    Jordan-Wigner mapped onto ``2 * n_sites`` qubits, ordered spin-block
    major: qubit index = spin * n_sites + site (spin 0 = up, spin 1 =
    down). With this ordering, intra-spin hopping is nearest-neighbor
    (no Jordan-Wigner string) and the on-site interaction is the only
    non-adjacent two-qubit gate — handled automatically by
    :class:`~mps_xtrap.circuits.MPSSimulator`'s swap network.

    Parameters
    ----------
    dt : float
        Trotter step size.
    n_sites : int
        Number of lattice sites (uses ``2 * n_sites`` qubits).
    t_total : float
        Total evolution time.
    t_hop : float
        Nearest-neighbor hopping amplitude.
    U : float
        On-site interaction strength.
    mu : float
        Chemical potential.
    boundary : {'open', 'periodic'}
    order : {1, 2}
    """
    _validate_boundary(boundary)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    n_qubits = 2 * n_sites
    bonds = _bonds(n_sites, boundary)

    c = Circuit(n_qubits)

    def hop_layer(sub_dt: float) -> None:
        theta = -t_hop * sub_dt
        for spin in (0, 1):
            off = spin * n_sites
            for (i, j) in bonds:
                qi, qj = off + i, off + j
                c.xx(theta, qi, qj)
                c.yy(theta, qi, qj)

    def onsite_layer(sub_dt: float) -> None:
        if U == 0.0:
            return
        for i in range(n_sites):
            q_up, q_dn = i, n_sites + i
            c.rz(-U * sub_dt / 2.0, q_up)
            c.rz(-U * sub_dt / 2.0, q_dn)
            c.zz(U * sub_dt / 2.0, q_up, q_dn)

    def mu_layer(sub_dt: float) -> None:
        if mu == 0.0:
            return
        for q in range(n_qubits):
            c.rz(mu * sub_dt, q)

    for _ in range(_n_steps(dt, t_total)):
        if order == 1:
            hop_layer(dt)
            onsite_layer(dt)
            mu_layer(dt)
        else:
            hop_layer(dt / 2)
            onsite_layer(dt / 2)
            mu_layer(dt)
            onsite_layer(dt / 2)
            hop_layer(dt / 2)

    return c


# ---------------------------------------------------------------------------
# Model registry + automatic dt-only binding for TrotterExtrapolator
# ---------------------------------------------------------------------------

MODEL_BUILDERS: Dict[str, Callable[..., Circuit]] = {
    "tfim": tfim_circuit,
    "heisenberg": heisenberg_circuit,
    "h2": h2_circuit,
    "fermi_hubbard": fermi_hubbard_circuit,
}


def bind_circuit(model: Union[str, Callable[..., Circuit]], /, **kwargs: Any) -> Callable[[float], Circuit]:
    """
    Bind every parameter of a model builder *except* ``dt``, returning a
    plain ``dt -> Circuit`` callable — exactly the signature
    :class:`~mps_xtrap.extrapolation.TrotterExtrapolator` and
    :class:`~mps_xtrap.extrapolation.TrotterSweepConfig` expect, so a
    Trotter dt-sweep "just works" without hand-written
    ``functools.partial`` calls.

    Parameters
    ----------
    model : str or callable
        One of ``'tfim'``, ``'heisenberg'``, ``'h2'``, ``'fermi_hubbard'``,
        or a circuit-builder function with the same ``(dt, ...)`` shape.
    **kwargs
        Every other keyword argument the builder needs (n_qubits/n_sites,
        t_total, couplings, boundary, order, ...).

    Examples
    --------
    >>> from mps_xtrap import bind_circuit, TrotterExtrapolator
    >>> circuit_fn = bind_circuit('tfim', n_qubits=8, t_total=1.0, J=1.0, h=0.5)
    >>> extrap = TrotterExtrapolator(order=2, chi=64)
    >>> result = extrap.run(circuit_fn, dt_values=[0.2, 0.1, 0.05],
    ...                      observables={'Z0': ('Z', 0)})

    See the README's "Driving TrotterExtrapolator yourself" section for
    the recommended :class:`~mps_xtrap.extrapolation.TrotterSweepConfig`
    version of this.
    """
    fn = MODEL_BUILDERS[model] if isinstance(model, str) else model
    return functools.partial(fn, **kwargs)


# ---------------------------------------------------------------------------
# Observable specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservableSpec:
    """
    A named observable request: a Pauli string and the site(s) it acts on.

    Accepts the same shorthand as the rest of the library —
    ``('Z', 0)`` for a single site, ``('ZZ', 0, 3)`` for a two-site
    correlator, or ``('XYZ', [0, 2, 5])`` for an N-site correlator.
    """
    pauli: str
    sites: Tuple[int, ...]

    @classmethod
    def parse(cls, spec: Union["ObservableSpec", Tuple]) -> "ObservableSpec":
        if isinstance(spec, ObservableSpec):
            return spec
        pauli = spec[0]
        rest = spec[1:]
        if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
            sites = tuple(rest[0])
        else:
            sites = tuple(rest)
        return cls(pauli=pauli, sites=sites)

    def evaluate(self, sim: MPSSimulator, state) -> float:
        if len(self.sites) == 1:
            return sim.expectation_value(state, self.pauli, self.sites[0])
        if len(self.sites) == 2:
            return sim.expectation_value(state, self.pauli, self.sites[0], self.sites[1])
        return sim.expectation_correlator(state, self.pauli, list(self.sites))


ObservableInput = Dict[str, Union[Tuple, ObservableSpec]]


def _normalize_observables(observables: ObservableInput) -> Dict[str, ObservableSpec]:
    return {name: ObservableSpec.parse(spec) for name, spec in observables.items()}


# ---------------------------------------------------------------------------
# Result type — pick your verbosity with ResultMode
# ---------------------------------------------------------------------------

class ResultMode(Enum):
    """
    Controls what :func:`simulate_hamiltonian` hands back.

    VALUES
        Plain numbers: a single float if you asked for one observable, or
        a ``{name: float}`` dict if you asked for several.
    FULL
        A :class:`HamiltonianResult` carrying the values plus simulation
        metadata (truncation error, bond dimensions, circuit stats, the
        parameters used, ...).
    """
    VALUES = "values"
    FULL = "full"


@dataclass
class HamiltonianResult:
    """Outcome of a single-``dt`` Hamiltonian-circuit simulation."""
    values: Dict[str, float]
    dt: float
    n_qubits: int
    circuit_depth: int
    two_qubit_count: int
    truncation_error: float
    bond_dimensions: List[int]
    max_bond_dim: int
    model: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> Union[float, Dict[str, float]]:
        """The lone observable value if only one was requested, else the full dict."""
        if len(self.values) == 1:
            return next(iter(self.values.values()))
        return dict(self.values)

    def summary(self) -> str:
        lines = [
            f"Hamiltonian simulation: model={self.model}, n_qubits={self.n_qubits}, dt={self.dt:g}",
            f"  Circuit: depth={self.circuit_depth}, two_qubit_gates={self.two_qubit_count}",
            f"  Truncation error: {self.truncation_error:.3e}  (max bond dim: {self.max_bond_dim})",
            "  Observables:",
        ]
        for name, val in self.values.items():
            lines.append(f"    {name} = {val:.8f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def simulate_hamiltonian(
    model: Union[str, Callable[..., Circuit]],
    /,
    dt: float,
    observables: ObservableInput,
    chi: int = 64,
    svd_threshold: float = 1e-12,
    device: str = "cpu",
    fusion: Union[bool, Any] = False,
    result_mode: ResultMode = ResultMode.FULL,
    **model_kwargs: Any,
) -> Union[float, Dict[str, float], HamiltonianResult]:
    """
    Build the requested Hamiltonian's Trotter circuit at a single ``dt``,
    simulate it, and read off the requested observable(s).

    For a dt-sweep with Richardson extrapolation to the continuum limit,
    use :func:`bind_circuit` together with
    :class:`~mps_xtrap.extrapolation.TrotterExtrapolator` /
    :class:`~mps_xtrap.extrapolation.TrotterSweepConfig` directly — see
    the README.

    Parameters
    ----------
    model : str or callable
        ``'tfim'``, ``'heisenberg'``, ``'h2'``, ``'fermi_hubbard'``, or a
        custom ``(dt, ...) -> Circuit`` builder.
    dt : float
        Trotter step size for this run.
    observables : dict
        ``{name: (pauli_string, site_or_sites)}``, e.g.
        ``{'Z0': ('Z', 0), 'ZZ03': ('ZZ', 0, 3)}``.
    chi, svd_threshold, device, fusion :
        Passed straight to :class:`~mps_xtrap.circuits.MPSSimulator`.
    result_mode : ResultMode
        ``ResultMode.VALUES`` for plain numbers, ``ResultMode.FULL`` (the
        default) for a :class:`HamiltonianResult` with metadata.
    **model_kwargs
        Every other keyword the chosen model builder needs (n_qubits /
        n_sites, t_total, couplings, boundary, order, ...).

    Returns
    -------
    float, dict, or HamiltonianResult
        Shape depends on ``result_mode`` and the number of observables
        requested (see :attr:`HamiltonianResult.value`).
    """
    fn = MODEL_BUILDERS[model] if isinstance(model, str) else model
    model_name = model if isinstance(model, str) else getattr(model, "__name__", "custom")

    circuit = fn(dt, **model_kwargs)
    sim = MPSSimulator(chi=chi, svd_threshold=svd_threshold, device=device, fusion=fusion)
    state = sim.run(circuit)

    specs = _normalize_observables(observables)
    values = {name: spec.evaluate(sim, state) for name, spec in specs.items()}

    if result_mode == ResultMode.VALUES:
        return values[next(iter(values))] if len(values) == 1 else values

    result = HamiltonianResult(
        values=values,
        dt=dt,
        n_qubits=circuit.n,
        circuit_depth=circuit.depth(),
        two_qubit_count=circuit.two_qubit_count(),
        truncation_error=state.total_truncation_error(),
        bond_dimensions=state.bond_dimensions(),
        max_bond_dim=state.max_bond_dim(),
        model=model_name,
        model_kwargs=dict(model_kwargs),
    )
    return result
