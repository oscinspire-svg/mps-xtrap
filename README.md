# mps_xtrap — Fast MPS Quantum Circuit Simulator with Gate Fusion

A production-ready quantum circuit simulator based on Matrix Product States (MPS),
built around **gate fusion** as its headline performance feature — with full GPU
acceleration via CuPy, and **qubit measurement** via projector operators.

Tensor networks (MPS) and gate fusion are a natural pairing, and together they
sell mps_xtrap on three fronts:

- **Larger qubit counts.** The MPS representation itself is what makes
  simulating large numbers of qubits tractable in the first place — circuits
  that would blow up a dense statevector simulator fit comfortably in an MPS
  as long as entanglement stays bounded.
- **Speed — the main event.** Gate fusion attacks the specific operation — the
  per-gate SVD — that dominates MPS runtime. Fusing k consecutive gates on the
  same qubit(s) into one gate turns k SVD truncations into 1, which is where
  most of the wall-clock time goes. This is a direct, compounding speedup on
  top of what the tensor-network representation already buys you, and it's
  most pronounced for circuits with structured, repeated gate patterns
  (variational ansatz layers, Trotterized time evolution, QAOA layers). That
  same reduction in SVD steps is also what makes running at **larger bond
  dimensions** (chi) tractable in practice — fewer, cheaper truncations mean
  you can push chi higher for the same time budget, so the tensor network
  itself becomes both faster to run and able to capture more entanglement
  before cost becomes prohibitive.
- **Reduced truncation error, as a bonus.** Fewer SVD truncations also means
  fewer chances to accumulate truncation error along the way — a welcome side
  effect of fusing gates, though speed is still the point.

---

## Licensing

mps_xtrap is released under the **Business Source License 1.1 (BUSL-1.1)**.

- **Non-commercial use** — free to use under the terms of the BUSL-1.1.
- **Commercial use** — requires a separate commercial license agreement.
  Please contact [oscinspire@gmail.com](mailto:oscinspire@gmail.com) to arrange one.
- **Change Date** — on 2029-01-01 the Licensed Work converts to the MIT License.

Sponsorships and donations are welcomed at:
[https://flutterwave.com/pay/6yptuvqab8cu](https://flutterwave.com/pay/6yptuvqab8cu)

---

## Installation

```bash
pip install numpy            # required
pip install cupy-cuda12x     # optional: GPU support (match your CUDA version)
pip install mps-xtrap
```

---

## Quick Start

```python
from mps_xtrap import Circuit, MPSSimulator

# --- Simple simulation ---
c = Circuit(4)
c.h(0).cx(0, 1).cx(1, 2).cx(2, 3)   # GHZ state

sim = MPSSimulator(chi=64)
state = sim.run(c)
print(state.expectation_pauli_z(0))   # -> 0.0

# --- GPU simulation ---
sim_gpu = MPSSimulator(chi=64, device='cuda')
state_gpu = sim_gpu.run(c)

# --- With gate fusion (the fast path) ---
sim_fused = MPSSimulator(chi=64, fusion=True)
state = sim_fused.run(c)   # same result, fewer SVDs, less wall-clock time

# --- Correlated-system expectation values (not single-site!) ---
# Two-site correlator <Z0 Z3>, sites need not be adjacent:
print(sim.expectation_value(state, 'ZZ', site=0, site2=3))
# Equivalent, and generalizes to any number of sites:
print(sim.expectation_correlator(state, 'ZZ', sites=[0, 3]))
# N-site correlator <Z0 X2 Y5>:
print(sim.expectation_correlator(state, 'ZXY', sites=[0, 2, 5]))

# --- Qubit measurement (v2.2.0) ---
from mps_xtrap import measure_qubit, measure_qubits, sample_counts, full_distribution

mres = measure_qubit(state, site=0)
print(mres.summary())              # P(0)=..., P(1)=...

jres = measure_qubits(state, sites=[0, 1, 2])
print(jres.summary())              # joint probability table

sc = sample_counts(state, sites=[0, 1], shots=1024, seed=42)
print(sc.summary())                # shot histogram

dist = full_distribution(state)    # {bitstring: probability}, n <= 20 only
```

---

## Architecture

```
mps_xtrap/
├── core/
│   └── mps.py              # MPS tensor train, SVD truncation, GPU/CPU backend
├── gates/
│   └── __init__.py         # Full gate library (H, CNOT, Rx, ZZ, ...)
├── circuits/
│   └── __init__.py         # Circuit builder API + MPSSimulator engine
├── fusion/
│   └── __init__.py         # Gate fusion (GateFuser, fuse) — the speed layer, added 2.1.0
├── measurement/
│   └── __init__.py         # Projector measurement + sampling    <- added 2.2.0
├── examples/
│   └── examples.py         # Runnable examples
└── cli.py                  # Command-line interface
```

---

## Gate Fusion — Tensor Networks, Made Fast

This is the core performance story of mps_xtrap. MPS simulation cost is
dominated by the SVD truncation each two-qubit gate triggers; gate fusion
merges consecutive gates on the **same qubit(s)** into a single combined gate
before simulation runs, so a chain of *k* gates costs one SVD instead of *k*.
For circuits with repeated structure — variational ansatz layers, Trotter
steps, QAOA layers — this is a direct, compounding speedup on top of what the
tensor-network representation already buys you over dense statevector
simulation.

```python
from mps_xtrap import MPSSimulator, GateFuser
from mps_xtrap.fusion import fuse

# At construction — the fast default
sim = MPSSimulator(chi=64, fusion=True)

# Fine-grained control
fuser = GateFuser(fuse_single=True, fuse_two=True, max_window=4)
sim = MPSSimulator(chi=64, fusion=fuser)

# Replace after construction
sim.fusion = GateFuser(fuse_single=False, fuse_two=True)

# Fuse a circuit directly and see the savings
fc = fuse(c)
report = GateFuser().fusion_report(c)
# {'original_count': 20, 'fused_count': 11, 'saved': 9, ...}
```

As a bonus, fusion also means fewer truncation errors along the way, not just
less compute: it's mathematically exact. Single-qubit fusion is just matrix
multiplication before contraction; two-qubit fusion of gates on the same pair
is the same operator applied once instead of several times. So the speedup
comes with no new truncation error stacked on top of whatever `chi` you're
already running at — but the speed is the point; the accuracy is free upside.

---

## Qubit Measurement

### Theory

A projective measurement on qubit `s` for outcome `k in {0, 1}` uses the projector:

```
P_k = |k><k|

P_0 = [[1, 0],   (projects onto |0>)
       [0, 0]]

P_1 = [[0, 0],   (projects onto |1>)
       [0, 1]]
```

The Born-rule probability of outcome `k` at site `s` is:

```
prob(k, s) = <psi| P_k^(s) |psi>
```

This is a single-site expectation value of the projector — computed efficiently
using the MPS canonicalization machinery already in the simulator.
Post-measurement collapse applies `P_k` as a gate to a copy of the MPS and
renormalizes the tensor at site `s` by `1/sqrt(prob(k,s))`.

### Single-qubit measurement

```python
from mps_xtrap import measure_qubit

res = measure_qubit(state, site=2)
print(res.prob0, res.prob1)      # Born-rule probabilities (always sum to 1)

# With collapse: returns the normalized post-measurement MPS for each outcome
res = measure_qubit(state, site=2, collapse=True)
state_if_0 = res.post_state_0   # MPS conditioned on observing |0>
state_if_1 = res.post_state_1   # MPS conditioned on observing |1> (None if prob=0)
```

### Multi-qubit joint measurement

```python
from mps_xtrap import measure_qubits

# Joint probability via chain rule: P(b0,b1,...) = P(b0)*P(b1|b0)*...
res = measure_qubits(state, sites=[0, 1, 2])
print(res.probabilities)            # {(0,0,0): p, (0,0,1): p, ...}
print(res.marginals[0].prob0)       # marginal P(0) at site 0
print(res.most_likely_bitstring())  # e.g. (0, 0, 1)
print(res.summary())                # formatted table
```

### Simulated shot sampling

```python
from mps_xtrap import sample_counts

sc = sample_counts(state, sites=[0, 1, 2, 3], shots=1024, seed=42)
print(sc.counts)            # {'0000': 512, '1111': 512, ...}
print(sc.probabilities())   # normalize to [0, 1]
print(sc.summary())
```

### Full probability distribution (small systems)

```python
from mps_xtrap import full_distribution

dist = full_distribution(state)   # {'0000': 0.5, '1111': 0.5, ...}
# Only feasible for n <= 20 qubits
```

### Projector constants

```python
from mps_xtrap import projector, P0, P1

P0 = projector(0)   # [[1,0],[0,0]]
P1 = projector(1)   # [[0,0],[0,1]]

# P0 and P1 can be passed directly to MPS.expectation_single()
prob0 = float(state.expectation_single(P0, site=2).real)
```

---

## GPU Support

```python
sim = MPSSimulator(chi=128, device='cuda')
state = sim.run(circuit)

state_gpu = state.to('cuda')
state_cpu = state_gpu.to('cpu')
```

Automatic CPU fallback if CuPy is missing or no GPU is detected.
Most beneficial at large bond dimensions (chi >= 64).

---

## Gate Library

**Single-qubit:** `I, X, Y, Z, H, S, T, Sdg, Tdg, Rx(theta), Ry(theta), Rz(theta), P(phi), U(theta,phi,lambda)`

**Two-qubit:** `CNOT/CX, CZ, SWAP, iSWAP, XX(theta), YY(theta), ZZ(theta), CRz(theta), CP(phi)`

---

## Circuit Builder API

```python
import numpy as np
from mps_xtrap import Circuit, MPSSimulator

c = Circuit(6)
c.h(0).cx(0, 1).rz(np.pi/4, 2).zz(0.5, 3, 4).swap(4, 5)

state = MPSSimulator(chi=64).run(c)
print(state.expectation_pauli_z(0))
print(state.bond_dimensions())
print(state.total_truncation_error())
```

---

## API Reference

### `MPSSimulator(chi, svd_threshold=1e-14, device='cpu', fusion=False)`
- `.run(circuit) -> MPS`
- `.run_from(circuit, state) -> MPS`
- `.expectation_value(state, observable, site, site2=None) -> float`
  Single-site by default (e.g. `'Z'`). Pass `site2` for a two-site
  correlator (e.g. `observable='ZZ'`, `site`, `site2`); the two sites
  need not be adjacent.
- `.expectation_correlator(state, observable, sites) -> float`
  N-site Pauli-string correlator for any number of sites, adjacent or
  not, e.g. `expectation_correlator(state, 'ZXZ', [0, 2, 5])`.
  `len(observable)` must match `len(sites)`.
- `.fusion` — read/replace the active fuser

### `GateFuser(fuse_single=True, fuse_two=True, max_window=None)`
- `__call__(circuit_or_instructions)` -> fused `Circuit` or list
- `.fusion_report(circuit) -> dict`

### `fuse(circuit, ...) -> Circuit`

### `MPS`
- `.expectation_pauli_z/x/y(site) -> float` — single-site only
- `.expectation_single(op, site) -> complex` — single-site only, arbitrary 2x2 operator
- `.expectation_two(op, site_i) -> complex` — two-site operator on the
  *adjacent* pair `(site_i, site_i + 1)`
- `.expectation_two_site(op_i, op_j, site_i, site_j) -> complex` —
  two-site correlator `<op_i(site_i) op_j(site_j)>` for arbitrary,
  possibly non-adjacent sites
- `.expectation_multi_site(ops, sites) -> complex` — the general,
  correlated-system building block: `<op_0(sites[0]) op_1(sites[1]) ...>`
  for any number of single-site operators at arbitrary, possibly
  non-adjacent sites, computed in a single O(n · chi^3) sweep.
  `expectation_two_site` is a thin wrapper around this.
- `.to_statevector() -> ndarray`  (n <= 20)
- `.bond_dimensions() -> list`, `.total_truncation_error() -> float`
- `.to(device) -> MPS`, `.copy() -> MPS`

> **Single-site vs. correlated systems:** `expectation_pauli_z/x/y` and
> `expectation_single` only ever act on one site and say nothing about
> correlations between qubits. To get expectation values *of the
> correlated system* — e.g. `<Z0 Z3>`, `<X1 Y4 Z7>` — use
> `expectation_two_site` / `expectation_multi_site` on `MPS` directly,
> or the higher-level `MPSSimulator.expectation_value(..., site2=...)`
> / `MPSSimulator.expectation_correlator(...)` wrappers above.

### `projector(k) -> ndarray`
Returns `|k><k|` for `k in {0, 1}`.

### `P0`, `P1`
Module-level projector constants.

### `MeasurementEngine(state)`
- `.measure_qubit(site, collapse=False) -> SingleMeasurementResult`
- `.measure_qubits(sites) -> MultiQubitMeasurementResult`
- `.joint_probability(sites, outcomes) -> float`
- `.sample_counts(sites, shots, seed) -> SampledCounts`
- `.full_distribution() -> dict`

### `measure_qubit(state, site, collapse=False) -> SingleMeasurementResult`
- `.prob0`, `.prob1`, `.post_state_0`, `.post_state_1`
- `.most_likely() -> int`, `.summary() -> str`

### `measure_qubits(state, sites) -> MultiQubitMeasurementResult`
- `.probabilities`, `.marginals`, `.most_likely_bitstring()`, `.summary()`

### `sample_counts(state, sites, shots=1024, seed=None) -> SampledCounts`
- `.counts`, `.probabilities()`, `.shots`, `.summary()`

### `full_distribution(state) -> dict`
Full `{bitstring: probability}` for n <= 20.

### `show_license() -> None`
Prints the mps_xtrap licensing notice.

---

## CLI

```bash
mps_xtrap simulate  --circuit ghz   --n 10 --chi 64
mps_xtrap benchmark --circuit ising --n 8  --chi-start 8 --chi-levels 4
mps_xtrap info
```

---

## Limitations

- Non-adjacent two-qubit gates use SWAP chains (increases depth)
- `to_statevector()` and `full_distribution()` only for n <= 20
- `measure_qubits()` enumerates 2^m bitstrings — only for m <= 20; use `sample_counts` for large m
- Two-qubit fusion only merges gates on the exact same ordered qubit pair — interleaved gates on other pairs act as a break
- GPU requires NVIDIA hardware and matching CuPy

---

## Dependencies

- Python 3.8+
- NumPy >= 1.21
- CuPy (optional, for GPU)

---
