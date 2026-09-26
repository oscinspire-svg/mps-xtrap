"""
mps_xtrap — Fast MPS-based quantum circuit simulator with overlapping-
support gate fusion and Trotter-step Richardson extrapolation.

Gate fusion is the headline performance feature: fusing runs of consecutive
gates that share qubit support — not just gates on the exact same qubit or
qubit pair — into single combined gates before simulation collapses many
SVD truncations into one, which is where most MPS simulation time goes.
Combined with the tensor-network (MPS) representation itself, this makes
simulation of structured circuits (ansatz layers, Trotter steps, QAOA
layers) significantly faster — with no added truncation error, since
fusion is mathematically exact.

On top of that, mps_xtrap ships Trotter-step Richardson extrapolation:
run a Trotterized time-evolution circuit at a few step sizes ``dt`` and
extrapolate observables to the ``dt -> 0`` continuum limit, using the
well-established ``O(dt^p)`` error order of the Trotter/Suzuki product
formula that built the circuit.

Quick start
-----------
>>> from mps_xtrap import Circuit, MPSSimulator
>>>
>>> # Simple simulation at a fixed bond dimension
>>> c = Circuit(4)
>>> c.h(0).cx(0,1).cx(1,2).cx(2,3)
>>> sim = MPSSimulator(chi=64)
>>> state = sim.run(c)
>>> print(state.expectation_pauli_z(0))
>>>
>>> # With gate fusion enabled — the fast path
>>> sim_fused = MPSSimulator(chi=64, fusion=True)
>>> state = sim_fused.run(c)
>>>
>>> # Trotter-step Richardson extrapolation
>>> from mps_xtrap import TrotterExtrapolator
>>> def trotter_circuit(dt, n=6, t_total=1.0):
...     steps = round(t_total / dt)
...     c = Circuit(n)
...     for _ in range(steps):
...         for q in range(n):
...             c.rx(0.5 * dt, q)
...         for q in range(n - 1):
...             c.zz(1.0 * dt, q, q + 1)
...     return c
>>> extrap = TrotterExtrapolator(order=2, chi=64)
>>> result = extrap.run(trotter_circuit, dt_values=[0.2, 0.1, 0.05],
...                      observables={'Z0': ('Z', 0)})
>>> print(result.summary())
>>>
>>> # Qubit measurement via projector operators
>>> from mps_xtrap import measure_qubit, measure_qubits, sample_counts
>>> mres = measure_qubit(state, site=0)
>>> print(mres.summary())
>>>
>>> # Two-qubit and N-site correlators, including non-adjacent sites
>>> print(sim.expectation_correlator(state, 'ZZ', sites=[0, 3]))
>>> print(sim.expectation_correlator(state, 'ZXY', sites=[0, 2, 5]))

Licensing
---------
mps_xtrap is dual-licensed.

  Non-commercial use  — free to use.
  Commercial use      — contact oscinspire@gmail.com for a commercial license.

Sponsorships and donations: https://flutterwave.com/pay/6yptuvqab8cu
"""

from .mps import MPS
from .circuit import Circuit
from .instruction import GateInstruction
from .simulator import MPSSimulator
from .extrapolation import (
    TrotterExtrapolator,
    trotter_extrapolate,
    TrotterSweepConfig,
    TrotterSweepResult,
    ExtrapolationResult,
    MultiObservableResult,
)
from .gates import get_gate
from .fusion import GateFuser, fuse
from .hamiltonians import (
    tfim_circuit,
    heisenberg_circuit,
    h2_circuit,
    h2_energy,
    fermi_hubbard_circuit,
    MODEL_BUILDERS,
    bind_circuit,
    ObservableSpec,
    ResultMode,
    HamiltonianResult,
    simulate_hamiltonian,
    H2_DEFAULT_COEFFICIENTS,
)
from .measurement import (
    projector,
    P0,
    P1,
    MeasurementEngine,
    SingleMeasurementResult,
    MultiQubitMeasurementResult,
    SampledCounts,
    measure_qubit,
    measure_qubits,
    sample_counts,
    full_distribution,
)
from .license import show_license

__version__ = "3.1.0"
__author__ = "MPS Sim"

__all__ = [
    # Core
    "MPS",
    "Circuit",
    "GateInstruction",
    "MPSSimulator",
    # Extrapolation (Trotter-step Richardson extrapolation)
    "TrotterExtrapolator",
    "trotter_extrapolate",
    "TrotterSweepConfig",
    "TrotterSweepResult",
    "ExtrapolationResult",
    "MultiObservableResult",
    # Gates
    "get_gate",
    # Fusion (overlapping/shared-qubit support fusion)
    "GateFuser",
    "fuse",
    # Hamiltonian simulation (predefined circuit builders)
    "tfim_circuit",
    "heisenberg_circuit",
    "h2_circuit",
    "h2_energy",
    "fermi_hubbard_circuit",
    "MODEL_BUILDERS",
    "bind_circuit",
    "ObservableSpec",
    "ResultMode",
    "HamiltonianResult",
    "simulate_hamiltonian",
    "H2_DEFAULT_COEFFICIENTS",
    # Measurement
    "projector",
    "P0",
    "P1",
    "MeasurementEngine",
    "SingleMeasurementResult",
    "MultiQubitMeasurementResult",
    "SampledCounts",
    "measure_qubit",
    "measure_qubits",
    "sample_counts",
    "full_distribution",
    # Licensing
    "show_license",
]
