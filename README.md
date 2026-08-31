# mps_xtrap; Matrix Product State (Tensor Network) Quantum Circuit Simulator with Overlapping-Support Gate Fusion

A production-ready quantum circuit simulator based on Matrix Product States (MPS),
built around **overlapping-support gate fusion** as its headline performance feature,
plus **Trotter-step Richardson extrapolation** for time-evolution circuits — with
full GPU acceleration via CuPy, and **qubit measurement** via projector operators.

Tensor networks (MPS) and gate fusion are a natural pairing, and together they
sell mps_xtrap on three fronts:

- **Larger qubit counts.** The MPS representation itself is what makes
  simulating large numbers of qubits tractable in the first place — circuits
  that would blow up a dense statevector simulator fit comfortably in an MPS
  as long as entanglement stays bounded.
- **Speed — the main event.** Fusing k consecutive gates that share qubit
  support into one dense gate means applying it to the MPS once instead of
  k times — one contraction instead of k small ones, far less data movement
  and intermediate-tensor allocation between steps, and fewer round trips
  through memory. Because fusion here follows *overlapping support* rather
  than requiring an exact same-qubit or same-pair match, it reaches
  patterns same-pair fusion structurally cannot — a CNOT staircase
  (`CX(0,1)`, `CX(1,2)`, `CX(2,3)`, ...) has no two gates on the same pair,
  yet every consecutive pair overlaps by one qubit, so the whole staircase
  collapses into one dense gate. This is most pronounced for circuits with
  structured, repeated gate patterns (variational ansatz layers, Trotterized
  time evolution, QAOA layers, swap networks).

On top of that, **Trotter-step Richardson extrapolation** targets a different
and well-established source of error: for circuits that Trotterize a
time-evolution operator, the error introduced by a finite step size `dt`
follows a known power law, `O(dt^p)`, fixed by which product formula (p=1
Lie-Trotter, p=2 Strang, ...) built the circuit. mps_xtrap runs the same
Trotterized circuit at a handful of step sizes and Richardson-extrapolates
the result to the `dt -> 0` continuum limit — the same idea Romberg
integration uses for numerical quadrature, applied to product-formula time
evolution.

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

# --- With overlapping-support gate fusion (the fast path) ---
sim_fused = MPSSimulator(chi=64, fusion=True)
state = sim_fused.run(c)   # same result, less overhead/data movement, less wall-clock time

# --- Correlated-system expectation values (not single-site!) ---
# Two-site correlator <Z0 Z3>, sites need not be adjacent:
print(sim.expectation_value(state, 'ZZ', site=0, site2=3))
# Equivalent, and generalizes to any number of sites:
print(sim.expectation_correlator(state, 'ZZ', sites=[0, 3]))
# N-site correlator <Z0 X2 Y5>:
print(sim.expectation_correlator(state, 'ZXY', sites=[0, 2, 5]))

# --- Trotter-step Richardson extrapolation ---
from mps_xtrap import TrotterExtrapolator, TrotterSweepConfig

def trotter_circuit(dt, n=6, J=1.0, h=0.5, t_total=1.0):
    """Second-order (Strang) Trotterized Ising evolution at step size dt."""
    steps = round(t_total / dt)
    c = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            c.rx(h * dt, q)
        for q in range(n - 1):
            c.zz(2 * J * dt, q, q + 1)
        for q in range(n):
            c.rx(h * dt, q)
    return c

extrapolator = TrotterExtrapolator(order=2, chi=64)
sweep = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)
result = extrapolator.run_sweep(
    trotter_circuit,
    sweep,
    observables={'Z0': ('Z', 0)},
)
print(result.summary())   # extrapolated <Z0> at dt -> 0, with uncertainty

# --- Qubit measurement ---
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
│   └── __init__.py         # Overlapping-support gate fusion (GateFuser, fuse)
├── extrapolation/
│   └── __init__.py         # Trotter-step Richardson extrapolation
├── measurement/
│   └── __init__.py         # Projector measurement + sampling
├── examples/
│   └── examples.py         # Runnable examples
└── cli.py                  # Command-line interface
```

---

## Gate Fusion — Overlapping (Shared-Qubit) Support

This is the core performance story of mps_xtrap. Applying gates to an MPS
one at a time means paying setup, contraction, and memory-access overhead
on every single gate. Fusing a chain of *k* gates that share qubit support
into one dense block collapses that into a single, larger operation applied
once — fewer intermediate contractions, less data shuffled in and out of
memory between steps, and one larger, more parallelism-friendly operation
in place of many small sequential ones.

Earlier fusion designs only merge gates on the *identical* qubit, or the
*identical* qubit pair. mps_xtrap fuses on **overlapping support** instead:
a run of consecutive gates merges into one block as long as each new gate
shares at least one qubit with the block built so far, up to a configurable
`max_qubits` cap. Same-qubit chains and same-pair runs fall out as the
special cases where the block's qubit set happens to stay at size 1 or 2 —
overlapping-support fusion is a strict generalization of both, and reaches
patterns they cannot:

```python
from mps_xtrap import Circuit, MPSSimulator, GateFuser
from mps_xtrap.fusion import fuse

c = Circuit(4)
c.cx(0, 1).cx(1, 2).cx(2, 3)   # a "CNOT staircase" — no two gates share a pair

report = GateFuser(max_qubits=2).fusion_report(c)
print(report)   # {'saved': 0, ...} — same-pair fusion finds nothing to merge

report = GateFuser(max_qubits=4).fusion_report(c)
print(report)   # {'saved': 2, 'blocks_by_size': {4: 1}, ...} — one 4-qubit fused gate
```

```python
# At construction — the fast default (max_qubits=3)
sim = MPSSimulator(chi=64, fusion=True)

# Fine-grained control
fuser = GateFuser(max_qubits=4, max_window=None)
sim = MPSSimulator(chi=64, fusion=fuser)

# Replace after construction
sim.fusion = GateFuser(max_qubits=2)

# Fuse a circuit directly and see the savings
fc = fuse(c, max_qubits=3)
report = GateFuser().fusion_report(c)
# {'original_count': 20, 'fused_count': 11, 'saved': 9, 'blocks_by_size': {...}}
```

### How it works

Each gate in a block is embedded into the joint operator space of the
block's full qubit set — acting as identity on any qubit in the block the
gate doesn't itself touch — and the embedded operators are multiplied
together in circuit order. The result is exactly the product of the
original gates: fusion never changes the mathematical operation, only how
many gate applications it takes to reach the same state.

Applying a fused block back to the MPS works the same way a normal
two-qubit gate does, generalized: if the block's qubits aren't already
adjacent, a SWAP network gathers them into contiguous positions, the block
is contracted with the fused gate, and the result is re-split into
individual MPS site tensors — then the same swaps run in reverse to restore
the original qubit layout. This is the natural k-site generalization of the
update every two-qubit gate already goes through.

The speedup comes entirely from *how the work gets there*: one big
contraction and one pass through memory in place of many small, serial
gate applications. Fusion is exact and changes nothing about the underlying
math — it only changes how many gate applications it takes to reach the
same state, which is where the overlapping-support strategy pays off: by
merging on shared qubits rather than requiring identical qubits or pairs,
it collapses far more of a circuit into fewer, larger blocks than same-pair
fusion ever could.

---

## Trotter-Step Richardson Extrapolation

### Theory

Simulating time evolution under `H = sum_j H_j` on a Trotterized circuit
approximates `e^{-iHt}` with a product formula applied step by step:

```
e^{-iHt}  ~  [ S(dt) ]^{t/dt}
```

For a p-th order product formula (p=1: Lie-Trotter; p=2: symmetric/Strang
splitting; p=4, 6, ...: higher-order Suzuki formulas), the error in an
observable `<O>` computed from the Trotterized circuit has the standard,
well-established form (Trotter 1959; Suzuki 1976):

```
<O>(dt)  ~  <O>(0) + c1 * dt^p + c2 * dt^(2p) + ...
```

Because the leading error order `p` is known from the product formula used
— not something fit from noisy data — Richardson extrapolation applies
directly and rigorously: simulate the same physical evolution (fixed total
time `T`, more Trotter steps as `dt` shrinks) at a handful of step sizes and
extrapolate to `dt -> 0`, cancelling one more power of `dt^p` at each level
— exactly how Romberg integration extrapolates numerical quadrature to zero
step size.

### Usage

Describe the sweep declaratively — a coarsest step size, a refinement
ratio, and a level count — rather than typing out a list of step sizes by
hand. This also splits the (expensive) simulation "collect" step from the
(cheap) Richardson-math "extrapolate" step, so you can re-extrapolate the
same collected data — e.g. at a different assumed `order` — without
re-simulating anything:

```python
from mps_xtrap import Circuit, TrotterExtrapolator, TrotterSweepConfig

def trotter_circuit(dt, n=8, J=1.0, h=0.5, t_total=1.0):
    steps = round(t_total / dt)
    c = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            c.rx(h * dt, q)              # half-step field (Strang splitting)
        for q in range(n - 1):
            c.zz(2 * J * dt, q, q + 1)   # full-step interaction
        for q in range(n):
            c.rx(h * dt, q)              # half-step field
    return c

sweep = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)

# Collect: runs trotter_circuit(dt) once per level
sweep_result = sweep(trotter_circuit, chi=64, observables={'Z0': ('Z', 0), 'ZZ_01': ('ZZ', 0, 1)})

# Extrapolate: cheap Richardson math on the already-collected data
extrapolator = TrotterExtrapolator(order=2, chi=64)
result = extrapolator.extrapolate(sweep_result)
print(result.summary())
print(result.observables['Z0'].extrapolated, result.observables['Z0'].uncertainty)

# One-shot convenience (collect + extrapolate together)
result = extrapolator.run_sweep(trotter_circuit, sweep, observables={'Z0': ('Z', 0)})
```

`circuit_fn(dt)` is responsible for holding the total physical evolution
time fixed across calls (e.g. `steps = round(t_total / dt)`) — the
extrapolation is only meaningful if every simulated circuit targets the
*same* underlying evolution.

`base_dt` is the *coarsest* (largest) step size; each level refines to a
step `refinement_ratio` times *smaller* than the one before.
`refinement_ratio` must be > 1 and `n_levels` must be >= 2.
`TrotterSweepConfig` is immutable (a frozen dataclass) — build a new one
rather than mutating an existing sweep's schedule.

### Reliability diagnostics

Every `ExtrapolationResult` carries:

- **Monotone-convergence check** — raw values should move consistently
  toward a limit as `dt` shrinks.
- **Correction-decay check** — successive Richardson corrections should
  shrink; if they don't, the extrapolation may not be trustworthy.
- **Uncertainty estimate** — from the size of the last correction applied.
- **Order-consistency check** — the assumed order `p` is compared against
  an order estimated directly from the data (log-log regression); a
  mismatch usually means the circuit doesn't actually implement the
  product formula you think it does.
- **MPS truncation-error check** — flags when bond-dimension truncation
  error isn't safely smaller than the Trotter correction being extracted,
  since in that regime you'd really be extrapolating MPS noise, not
  Trotter error. Choose `chi` large enough that this stays quiet across the
  whole `dt` sweep.

```python
print(result.observables['Z0'].is_reliable)
print(result.observables['Z0'].reliability_notes)
```

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

**Fused (3+ qubit):** produced automatically by `GateFuser` from overlapping
runs of the gates above — not hand-authored, but applied to the MPS exactly
like any other multi-qubit gate.

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

### `GateFuser(max_qubits=3, max_window=None)`
- `__call__(circuit_or_instructions)` -> fused `Circuit` or list
- `.fusion_report(circuit) -> dict` — `{'original_count', 'fused_count', 'saved', 'blocks_by_size'}`

### `fuse(circuit, max_qubits=3, max_window=None) -> Circuit`

### `TrotterExtrapolator(order=2, chi=64, device='cpu', verbose=True)`
- `.run(circuit_fn, dt_values, observables=None, measure=None) -> MultiObservableResult`
  Explicit `dt_values` list — collects and extrapolates in one call.
- `.extrapolate(sweep_result) -> MultiObservableResult`
  Richardson-extrapolates an already-collected `TrotterSweepResult` (no
  re-simulation) — cheap enough to call repeatedly, e.g. at different `order`.
- `.run_sweep(circuit_fn, config, observables=None, measure=None) -> MultiObservableResult`
  One-shot: collect a `TrotterSweepConfig` sweep and extrapolate it.

### `TrotterSweepConfig(base_dt, refinement_ratio, n_levels)`
Declarative step-size schedule; `dt_values[i] = base_dt / refinement_ratio**i`.
Frozen (immutable) dataclass.
- `.dt_values -> List[float]`
- `__call__(circuit_fn, chi=64, device='cpu', observables=None, measure=None, verbose=True) -> TrotterSweepResult`
  Runs `circuit_fn(dt)` once per level (`n_levels` runs total) — the
  "collect" phase. Does not extrapolate.

### `TrotterSweepResult`
Raw per-level results from calling a `TrotterSweepConfig`.
- `.config`, `.dt_values`, `.values` (dict of label -> list of raw values), `.truncation_errors`

### `trotter_extrapolate(circuit_fn, dt_values, order=2, chi=64, ...) -> MultiObservableResult`
Convenience one-off wrapper around `TrotterExtrapolator`.

### `ExtrapolationResult`
- `.dt_values`, `.raw_values`, `.extrapolated`, `.uncertainty`, `.table`
- `.order`, `.estimated_order`, `.max_truncation_error`
- `.is_reliable`, `.reliability_notes`
- `.summary() -> str`, `.improvement_factor() -> float`

### `MultiObservableResult`
- `.observables -> Dict[str, ExtrapolationResult]`, `.dt_values`
- `.summary() -> str`

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
- `.apply_block_svd(start, block_tensor, k, chi=None, svd_threshold=1e-14) -> float`
  Generalizes `.apply_svd_truncation` (the two-site case) to a k-site
  block — the mechanism fused gates from `GateFuser` use to re-split back
  into the MPS.
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
mps_xtrap simulate    --circuit ghz --n 10 --chi 64
mps_xtrap extrapolate --n 8 --dt 0.2 0.1 0.05 0.025 --order 2 --chi 64
mps_xtrap benchmark   --circuit ising --n 8 --chi-start 8 --chi-levels 4
mps_xtrap info
```

---

## Limitations

- Non-adjacent gates (two-qubit or fused multi-qubit) use SWAP networks
  (increases depth)
- `to_statevector()` and `full_distribution()` only for n <= 20
- `measure_qubits()` enumerates 2^m bitstrings — only for m <= 20; use `sample_counts` for large m
- Fusion blocks can grow the dense fused matrix to `2**max_qubits` square —
  costs rise quickly past `max_qubits` ~ 4-5; a gate that doesn't overlap
  the current block always closes it, so unrelated interleaved gates limit
  how far a block can grow regardless of `max_qubits`
- Trotter-step extrapolation assumes `circuit_fn` holds the total physical
  evolution time fixed across step sizes; it's the caller's responsibility
  to build `circuit_fn` that way
- GPU requires NVIDIA hardware and matching CuPy

---

## Dependencies

- Python 3.8+
- NumPy >= 1.21
- CuPy (optional, for GPU)

---
