# mps_xtrap: MPS Quantum Circuit Simulator with Trotter-Step Richardson Extrapolation

mps_xtrap is a quantum circuit simulator built on Matrix Product States (MPS). Its core feature is Trotter-step Richardson extrapolation: you run a Trotterized time-evolution circuit at several step sizes and extrapolate the results to the `dt -> 0` continuum limit, using the known `O(dt^p)` error order of the product formula. This is the same principle Romberg integration uses for numerical quadrature, applied here to product-formula time evolution.

The MPS representation is what makes large qubit counts practical in the first place. On top of that, mps_xtrap adds overlapping-support gate fusion for lower per-gate overhead, GPU acceleration through CuPy, and qubit measurement through projector operators.

## Why use it

**Better accuracy for the same simulation cost.** A single Trotterized run at one `dt` always carries Trotter error. Shrinking `dt` reduces that error but increases the number of steps and the total work. Richardson extrapolation takes a different approach: it simulates the same physical evolution (fixed total time `T`) at several step sizes, then uses the known error order `p` of the product formula to cancel the leading error term. The extrapolated estimate is typically more accurate than even the finest individually simulated `dt`, without requiring an impractically small step size. Reliability diagnostics (monotone convergence, correction decay, order consistency, and MPS truncation error checks) are computed alongside every result.

**Larger qubit counts.** Because the state is stored as an MPS instead of a dense statevector, circuits with bounded entanglement can be simulated at qubit counts that would overwhelm a dense simulator, which makes running the same circuit at multiple step sizes affordable even for large systems.

**Overlapping-support gate fusion.** mps_xtrap can merge runs of gates that share any qubit support, not just gates on the exact same qubit or pair, into single combined gates before simulation. This reduces the per-gate contraction and memory overhead that accumulates over a long circuit. Fusion is exact: it never changes simulation accuracy or the underlying SVD truncation. See "Gate Fusion" below for details.

---

## Licensing

mps_xtrap is released under the Business Source License 1.1 (BUSL-1.1).

- **Non-commercial use:** free under the terms of the BUSL-1.1.
- **Commercial use:** requires a separate commercial license. Contact [oscinspire@gmail.com](mailto:oscinspire@gmail.com) to arrange one.
- **Change date:** on 2029-01-01 the Licensed Work converts to the MIT License.

Sponsorships and donations are welcome at: [https://flutterwave.com/pay/6yptuvqab8cu](https://flutterwave.com/pay/6yptuvqab8cu)

---

## Installation

```bash
pip install numpy            # required
pip install cupy-cuda12x     # optional: GPU support (match your CUDA version)
```

---

## Quick Start

```python
from mps_xtrap import tfim_circuit, MPSSimulator

# Build a circuit with one of the predefined Hamiltonian builders.
# The other three are heisenberg_circuit, h2_circuit, and fermi_hubbard_circuit.
circuit = tfim_circuit(dt=0.05, n_qubits=6, t_total=1.0, J=1.0, h=0.5)

sim = MPSSimulator(chi=64)
state = sim.run(circuit)
print(state.expectation_pauli_z(0))

# GPU simulation
sim_gpu = MPSSimulator(chi=64, device='cuda')
state_gpu = sim_gpu.run(circuit)

# Correlated-system expectation values (not single-site)
# Two-site correlator <Z0 Z3>; sites need not be adjacent.
print(sim.expectation_value(state, 'ZZ', site=0, site2=3))
# Equivalent, and generalizes to any number of sites:
print(sim.expectation_correlator(state, 'ZZ', sites=[0, 3]))
# N-site correlator <Z0 X2 Y5>:
print(sim.expectation_correlator(state, 'ZXY', sites=[0, 2, 5]))

# Trotter-step Richardson extrapolation, driven by the same predefined builder.
from mps_xtrap import bind_circuit, TrotterExtrapolator, TrotterSweepConfig

circuit_fn = bind_circuit('tfim', n_qubits=6, t_total=1.0, J=1.0, h=0.5)  # dt -> Circuit

extrapolator = TrotterExtrapolator(order=2, chi=64)
sweep = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)
result = extrapolator.run_sweep(
    circuit_fn,
    sweep,
    observables={'Z0': ('Z', 0)},
)
print(result.summary())   # extrapolated <Z0> at dt -> 0, with uncertainty

# Overlapping-support gate fusion (lower per-gate overhead, same result)
sim_fused = MPSSimulator(chi=64, fusion=True)
state = sim_fused.run(circuit)

# Qubit measurement
from mps_xtrap import measure_qubit, measure_qubits, sample_counts, full_distribution

mres = measure_qubit(state, site=0)
print(mres.summary())              # P(0)=..., P(1)=...

jres = measure_qubits(state, sites=[0, 1, 2])
print(jres.summary())              # joint probability table

sc = sample_counts(state, sites=[0, 1], shots=1024, seed=42)
print(sc.summary())                # shot histogram

dist = full_distribution(state)    # {bitstring: probability}, n <= 20 only

# Any custom circuit can be built gate by gate with Circuit's fluent API.
# See "Circuit Builder API" below for the full gate set.
from mps_xtrap import Circuit

custom = Circuit(4)
custom.h(0).cx(0, 1).cx(1, 2).cx(2, 3)   # a GHZ state
custom_state = MPSSimulator(chi=64).run(custom)
print(custom_state.expectation_pauli_z(0))   # -> 0.0
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
├── extrapolation/
│   └── __init__.py         # Trotter-step Richardson extrapolation
├── hamiltonians/
│   └── __init__.py         # Predefined Hamiltonian circuits (TFIM, Heisenberg, H2, Fermi-Hubbard)
├── fusion/
│   └── __init__.py         # Overlapping-support gate fusion (GateFuser, fuse)
├── measurement/
│   └── __init__.py         # Projector measurement + sampling
├── examples/
│   └── examples.py         # Runnable examples
└── cli.py                  # Command-line interface
```

---

## Trotter-Step Richardson Extrapolation

### Theory

Simulating time evolution under `H = sum_j H_j` on a Trotterized circuit approximates `e^{-iHt}` with a product formula applied step by step:

```
e^{-iHt}  ~  [ S(dt) ]^{t/dt}
```

For a p-th order product formula (p=1 is Lie-Trotter; p=2 is symmetric Strang splitting; p=4, 6, ... are higher-order Suzuki formulas), the error in an observable `<O>` computed from the Trotterized circuit follows a known form (Trotter 1959; Suzuki 1976):

```
<O>(dt)  ~  <O>(0) + c1 * dt^p + c2 * dt^(2p) + ...
```

Because the leading error order `p` comes from the product formula itself, rather than being fit from data, Richardson extrapolation applies directly: simulate the same physical evolution (fixed total time `T`, more Trotter steps as `dt` shrinks) at a handful of step sizes, then extrapolate to `dt -> 0`, cancelling one more power of `dt^p` at each level. This is the same idea Romberg integration uses to extrapolate numerical quadrature to zero step size. The result is accuracy that would otherwise require an impractically small `dt`, obtained from a handful of cheaper, larger-`dt` runs.

### Usage

Describe the sweep declaratively: a coarsest step size, a refinement ratio, and a level count, rather than typing out a list of step sizes by hand. This also separates the expensive simulation "collect" step from the cheap Richardson-math "extrapolate" step, so you can re-extrapolate the same collected data (for example at a different assumed `order`) without re-simulating anything.

```python
from mps_xtrap import bind_circuit, TrotterExtrapolator, TrotterSweepConfig

# Any of the four predefined builders (tfim_circuit, heisenberg_circuit,
# h2_circuit, fermi_hubbard_circuit) works here. bind_circuit fixes every
# parameter except dt, returning the dt -> Circuit callable this API expects.
# See "Hamiltonian Simulation" below for each builder's full parameter list.
circuit_fn = bind_circuit('tfim', n_qubits=8, t_total=1.0, J=1.0, h=0.5)

sweep = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)

# Collect: runs circuit_fn(dt) once per level
sweep_result = sweep(circuit_fn, chi=64, observables={'Z0': ('Z', 0), 'ZZ_01': ('ZZ', 0, 1)})

# Extrapolate: cheap Richardson math on the already-collected data
extrapolator = TrotterExtrapolator(order=2, chi=64)
result = extrapolator.extrapolate(sweep_result)
print(result.summary())
print(result.observables['Z0'].extrapolated, result.observables['Z0'].uncertainty)

# One-shot convenience (collect + extrapolate together)
result = extrapolator.run_sweep(circuit_fn, sweep, observables={'Z0': ('Z', 0)})
```

`circuit_fn` does not have to come from `bind_circuit`. Any `dt -> Circuit` callable works, including one you write by hand gate by gate (see the custom-circuit example in Quick Start). Whichever it is, `circuit_fn(dt)` must hold the total physical evolution time fixed across calls (for example `steps = round(t_total / dt)`), since the extrapolation is only meaningful if every simulated circuit targets the same underlying evolution.

`base_dt` is the coarsest (largest) step size; each level refines to a step `refinement_ratio` times smaller than the one before. `refinement_ratio` must be greater than 1, and `n_levels` must be at least 2. `TrotterSweepConfig` is an immutable (frozen) dataclass, so build a new one rather than mutating an existing sweep's schedule.

### Reliability diagnostics

Every `ExtrapolationResult` carries:

- **Monotone convergence check:** raw values should move consistently toward a limit as `dt` shrinks.
- **Correction decay check:** successive Richardson corrections should shrink; if they do not, the extrapolation may not be trustworthy.
- **Uncertainty estimate:** derived from the size of the last correction applied.
- **Order consistency check:** the assumed order `p` is compared against an order estimated directly from the data (log-log regression). A mismatch usually means the circuit does not actually implement the product formula you think it does.
- **MPS truncation error check:** flags when bond-dimension truncation error is not safely smaller than the Trotter correction being extracted. In that regime you would effectively be extrapolating MPS noise rather than Trotter error, so choose `chi` large enough to keep this check quiet across the whole `dt` sweep.

```python
print(result.observables['Z0'].is_reliable)
print(result.observables['Z0'].reliability_notes)
```

---

## Hamiltonian Simulation (Predefined Circuits)

Four ready-made Trotterized time-evolution circuit builders are provided, one per model, so you don't have to hand-write the gate sequence for these standard benchmarks. System size, step size, total time, couplings, boundary conditions, and which observable to read out are all arguments; nothing about the problem instance is hardcoded.

| Model | Builder | Qubit count |
|---|---|---|
| Transverse-Field Ising | `tfim_circuit` | `n_qubits` (your choice) |
| Heisenberg (XXX/XXZ) | `heisenberg_circuit` | `n_qubits` (your choice) |
| Hydrogen molecule (H2) | `h2_circuit` | fixed at 2 (tapered representation) |
| Fermi-Hubbard | `fermi_hubbard_circuit` | `2 * n_sites` (spin up and down) |

### Conventions

```
TFIM:          H = -J * sum_<i,j> Z_i Z_j  -  h * sum_i X_i
Heisenberg:    H = sum_<i,j> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) - hz * sum_i Z_i
H2:            H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*Y0Y1 + g5*X0X1
               (2-qubit Bravyi-Kitaev-tapered qubit Hamiltonian. Defaults
               are the R=0.75 A values from O'Malley et al., Phys. Rev. X
               6, 031007 (2016), Table I. Pass your own g0..g5 for any
               other bond length or basis set.)
Fermi-Hubbard: H = -t_hop * sum_{i,sigma} (c^dag_{i,sigma} c_{i+1,sigma} + h.c.)
                   + U * sum_i n_{i,up} n_{i,down} - mu * sum_{i,sigma} n_{i,sigma}
               (Jordan-Wigner mapped. Qubits are ordered spin-block-major,
               qubit = spin * n_sites + site, so intra-spin hopping is a
               nearest-neighbor gate and the on-site U term is the only
               long-range one. MPSSimulator's swap network handles this
               automatically.)
```

Every builder shares the same shape: `dt` first, then the model's required parameters (`n_qubits`/`n_sites`, `t_total`), then physics keywords with sensible defaults, `boundary` (`'open'` or `'periodic'`, where applicable), and `order` (1 for first-order Lie-Trotter, 2 for second-order Strang splitting, matching the `order` used elsewhere in this library).

### Argument positions

`dt` is always argument position 0 in every builder. It is the only parameter whose position is part of the API contract, which is what lets `bind_circuit`, `TrotterExtrapolator`, and `TrotterSweepConfig` all call an unqualified `circuit_fn(dt)` regardless of which model it came from.

Every other parameter should be passed by keyword. The four builders do not share a common positional layout beyond `dt`, so relying on position past argument 0 is the one place a mistake could silently swap two values instead of raising an error. For reference, here is each builder's actual positional order:

| Position | `tfim_circuit` | `heisenberg_circuit` | `h2_circuit` | `fermi_hubbard_circuit` |
|---|---|---|---|---|
| 0 | `dt` | `dt` | `dt` | `dt` |
| 1 | `n_qubits` | `n_qubits` | `t_total` | `n_sites` |
| 2 | `t_total` | `t_total` | `g0=None` | `t_total` |
| 3 | `J=1.0` | `model='XXX'` | `g1=None` | `t_hop=1.0` |
| 4 | `h=1.0` | `J=1.0` | `g2=None` | `U=4.0` |
| 5 | `boundary='open'` | `delta=1.0` | `g3=None` | `mu=0.0` |
| 6 | `order=2` | `hz=0.0` | `g4=None` | `boundary='open'` |
| 7 | (none) | `boundary='open'` | `g5=None` | `order=1` |
| 8 | (none) | `order=2` | `order=2` | (none) |

One case worth calling out: `h2_circuit` has no `n_qubits`/`n_sites` parameter at all, since it is always exactly 2 qubits. Its position 1 is `t_total`, not a qubit count, unlike the other three builders. Calling `h2_circuit(dt, 8)` would silently set `t_total=8` instead of raising an error, which is exactly the kind of mistake keyword arguments avoid. All examples in this README pass every parameter after `dt` by keyword for that reason, and `bind_circuit` enforces it (see below).

```python
from mps_xtrap import tfim_circuit, heisenberg_circuit, h2_circuit, fermi_hubbard_circuit, MPSSimulator

circuit = tfim_circuit(dt=0.05, n_qubits=10, t_total=1.0, J=1.0, h=0.5,
                        boundary='open', order=2)
state = MPSSimulator(chi=64).run(circuit)
print(state.expectation_pauli_z(0))
```

> **Note:** every circuit here starts from `|0...0>`. For several of these models, that state is a symmetry-protected fixed point of the dynamics. For example, the isotropic XX+YY terms in the Heisenberg model conserve total magnetization, and H2's default X0X1/Y0Y1 coefficients are equal, which cancels their effect on `|00>`. In those cases you will see no dynamics at all, which is correct physics rather than a simulator quirk. TFIM is the exception, since its transverse field does not leave `|0...0>` invariant.

### Driving `TrotterExtrapolator`

For a `dt`-sweep with Richardson extrapolation to the continuum limit, `TrotterExtrapolator` handles the sweep directly (see "Trotter-Step Richardson Extrapolation" above), and `bind_circuit` is the bridge: it binds every one of a model's keyword arguments except `dt`, returning a plain `dt -> Circuit` callable that both extrapolator APIs expect.

```python
from mps_xtrap import bind_circuit

circuit_fn = bind_circuit('tfim', n_qubits=10, t_total=1.0, J=1.0, h=0.5)  # dt -> Circuit
```

**Preferred: `TrotterSweepConfig`.** Describe the step-size schedule declaratively (a coarsest `dt`, a refinement ratio, a level count) instead of typing out a list by hand, and get the collect/extrapolate split for free, so you can re-run the cheap Richardson math at a different assumed `order` on the same collected data without re-simulating anything.

```python
from mps_xtrap import TrotterExtrapolator, TrotterSweepConfig

sweep = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)
extrapolator = TrotterExtrapolator(order=2, chi=64)

result = extrapolator.run_sweep(circuit_fn, sweep, observables={'Z0': ('Z', 0)})
print(result.summary())
```

**Also available: an explicit `dt_values` list**, if you would rather name the exact step sizes yourself instead of a base/ratio/levels schedule:

```python
result = extrapolator.run(circuit_fn, dt_values=[0.2, 0.1, 0.05, 0.025],
                           observables={'Z0': ('Z', 0)})
```

`bind_circuit` also accepts a custom builder function in place of a model name string, as long as it has the same `(dt, ...)` shape as the four built-in builders.

**How `dt` is passed.** `TrotterSweepConfig.__call__` and `TrotterExtrapolator.run` both call `circuit_fn(dt)` with `dt` as the sole positional argument. `bind_circuit(model, **kwargs)` returns `functools.partial(fn, **kwargs)`, binding every one of the builder's other parameters by keyword. `bind_circuit`'s own signature is `(model, /, **kwargs)`, so it only accepts keyword arguments for everything after `model`, which enforces the "always use keywords past `dt`" guidance above. The `dt` slot is left unbound, so `circuit_fn(dt)` runs `fn(dt, **kwargs)`, with `dt` landing in argument position 0 as described above. The only way to break this is to also pass `dt` as one of `bind_circuit`'s keywords (for example `bind_circuit('tfim', dt=0.05, ...)`), which raises `TypeError: got multiple values for argument 'dt'` the moment the sweep calls `circuit_fn(dt)`, giving a clear failure instead of a silent mismatch.

### H2 ground-state energy (not time evolution)

The `g0*I` term in the H2 Hamiltonian is a global phase under time evolution and is skipped by `h2_circuit`. If you want the actual electronic energy of a state (for example a VQE-style check against a known FCI value, rather than dynamics), use `h2_energy` directly. Note that `|00>` is not generally the ground state of this tapered Hamiltonian; the true ground state is whatever superposition minimizes `<H>`, normally reached through a variational ansatz rather than by picking a basis state. `h2_energy` reports the honest expectation value for whatever state you give it:

```python
from mps_xtrap import h2_energy, MPSSimulator, Circuit

state = MPSSimulator(chi=8).run(Circuit(2))   # |00>
print(h2_energy(state))   # about 0.706 Ha: a basis state, not the ground state

# The true ground energy (e.g. from exact diagonalization or a VQE run)
# at the default R=0.75 A is about -1.1456 Ha, matching FCI/literature.
```

---

## Gate Fusion (Overlapping, Shared-Qubit Support)

Applying gates to an MPS one at a time means paying setup, contraction, and memory-access overhead on every gate. Fusing a chain of *k* gates that share qubit support into one dense block collapses that into a single, larger operation applied once: fewer intermediate contractions, less data moved in and out of memory between steps, and one larger, more parallelism-friendly operation in place of many small sequential ones.

Earlier fusion designs only merge gates on the identical qubit, or the identical qubit pair. mps_xtrap fuses on overlapping support instead: a run of consecutive gates merges into one block as long as each new gate shares at least one qubit with the block built so far, up to a configurable `max_qubits` cap. Same-qubit chains and same-pair runs are the special cases where the block's qubit set happens to stay at size 1 or 2. Overlapping-support fusion is a strict generalization of both and reaches patterns neither can:

```python
from mps_xtrap import Circuit, MPSSimulator, GateFuser
from mps_xtrap.fusion import fuse

c = Circuit(4)
c.cx(0, 1).cx(1, 2).cx(2, 3)   # a "CNOT staircase": no two gates share a pair

report = GateFuser(max_qubits=2).fusion_report(c)
print(report)   # {'saved': 0, ...}: same-pair fusion finds nothing to merge

report = GateFuser(max_qubits=4).fusion_report(c)
print(report)   # {'saved': 2, 'blocks_by_size': {4: 1}, ...}: one 4-qubit fused gate
```

```python
# At construction, the default is max_qubits=3
sim = MPSSimulator(chi=64, fusion=True)

# Fine-grained control
fuser = GateFuser(max_qubits=4, max_window=None)
sim = MPSSimulator(chi=64, fusion=fuser)

# Replace after construction
sim.fusion = GateFuser(max_qubits=2)

# Fuse a circuit directly and see the savings
fc = fuse(c, max_qubits=3)
report = GateFuser().fusion_report(c)   # GateFuser() defaults to max_qubits=3
print(report)
# {'original_count': 3, 'fused_count': 2, 'saved': 1, 'blocks_by_size': {3: 1}}
```

### How it works

Each gate in a block is embedded into the joint operator space of the block's full qubit set, acting as identity on any qubit in the block that the gate does not itself touch, and the embedded operators are multiplied together in circuit order. The result is exactly the product of the original gates: fusion never changes the mathematical operation, only how many gate applications it takes to reach the same state.

Applying a fused block back to the MPS follows the same pattern as a normal two-qubit gate, generalized to more qubits: if the block's qubits are not already adjacent, a SWAP network gathers them into contiguous positions, the block is contracted with the fused gate, the result is re-split into individual MPS site tensors, and the same swaps run in reverse to restore the original qubit layout. This is the natural k-site generalization of the update every two-qubit gate already goes through.

The speedup comes from how the work is organized: one large contraction and one pass through memory in place of many small, serial gate applications. Fusion is exact and changes nothing about the underlying math; by merging on shared qubits rather than requiring identical qubits or pairs, it collapses far more of a circuit into fewer, larger blocks than same-pair fusion can.

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

This is a single-site expectation value of the projector, computed efficiently using the MPS canonicalization machinery already in the simulator. Post-measurement collapse applies `P_k` as a gate to a copy of the MPS and renormalizes the tensor at site `s` by `1/sqrt(prob(k,s))`.

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
print(sc.probabilities())   # normalized to [0, 1]
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

Falls back to CPU automatically if CuPy is missing or no GPU is detected. GPU acceleration is most beneficial at large bond dimensions (chi >= 64).

---

## Gate Library

**Single-qubit:** `I, X, Y, Z, H, S, T, Sdg, Tdg, Rx(theta), Ry(theta), Rz(theta), P(phi), U(theta,phi,lambda)`

**Two-qubit:** `CNOT/CX, CZ, SWAP, iSWAP, XX(theta), YY(theta), ZZ(theta), CRz(theta), CP(phi)`

**Fused (3+ qubit):** produced automatically by `GateFuser` from overlapping runs of the gates above. These are not hand-authored, but are applied to the MPS exactly like any other multi-qubit gate.

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
- `.fusion`: read or replace the active fuser

### `TrotterExtrapolator(order=2, chi=64, device='cpu', verbose=True)`
- `.run(circuit_fn, dt_values, observables=None, measure=None) -> MultiObservableResult`
  Takes an explicit `dt_values` list; collects and extrapolates in one call.
- `.extrapolate(sweep_result) -> MultiObservableResult`
  Richardson-extrapolates an already-collected `TrotterSweepResult` with no
  re-simulation. Cheap enough to call repeatedly, e.g. at different `order`.
- `.run_sweep(circuit_fn, config, observables=None, measure=None) -> MultiObservableResult`
  One-shot: collects a `TrotterSweepConfig` sweep and extrapolates it.

### `TrotterSweepConfig(base_dt, refinement_ratio, n_levels)`
Declarative step-size schedule; `dt_values[i] = base_dt / refinement_ratio**i`.
An immutable (frozen) dataclass.
- `.dt_values -> List[float]`
- `__call__(circuit_fn, chi=64, device='cpu', observables=None, measure=None, verbose=True) -> TrotterSweepResult`
  Runs `circuit_fn(dt)` once per level (`n_levels` runs total): the
  "collect" phase. Does not extrapolate.

### `TrotterSweepResult`
Raw per-level results from calling a `TrotterSweepConfig`.
- `.config`, `.dt_values`, `.values` (dict of label to list of raw values), `.truncation_errors`

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

### Hamiltonian simulation

#### `tfim_circuit(dt, n_qubits, t_total, J=1.0, h=1.0, boundary='open', order=2) -> Circuit`
#### `heisenberg_circuit(dt, n_qubits, t_total, model='XXX', J=1.0, delta=1.0, hz=0.0, boundary='open', order=2) -> Circuit`
`model='XXZ'` sets `Jz = delta * J` (`Jx = Jy = J`); `model='XXX'` sets `Jx = Jy = Jz = J`.

#### `h2_circuit(dt, t_total, g0=None, ..., g5=None, order=2) -> Circuit`
Always 2 qubits. Unset `g*` values fall back to `H2_DEFAULT_COEFFICIENTS`.

#### `h2_energy(state, g0=None, ..., g5=None, include_nuclear_repulsion=True) -> float`
Static `<H>` (including nuclear repulsion by default) for a given 2-qubit state, independent of any time evolution.

#### `fermi_hubbard_circuit(dt, n_sites, t_total, t_hop=1.0, U=4.0, mu=0.0, boundary='open', order=1) -> Circuit`
Uses `2 * n_sites` qubits; qubits `0..n_sites-1` are spin-up, `n_sites..2*n_sites-1` are spin-down.

#### `H2_DEFAULT_COEFFICIENTS -> dict`
`{'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'nuclear_repulsion', 'bond_length_angstrom'}`
at R=0.75 A (O'Malley et al. 2016).

#### `MODEL_BUILDERS -> dict`
`{'tfim': tfim_circuit, 'heisenberg': heisenberg_circuit, 'h2': h2_circuit, 'fermi_hubbard': fermi_hubbard_circuit}`

#### `bind_circuit(model, **kwargs) -> Callable[[float], Circuit]`
Binds every keyword except `dt`, for use with `TrotterExtrapolator`/`TrotterSweepConfig`.
`model` is a name from `MODEL_BUILDERS` or any `(dt, ...) -> Circuit` callable.

#### `ObservableSpec(pauli, sites)`
- `.parse(spec) -> ObservableSpec`: accepts `('Z', 0)`, `('ZZ', 0, 3)`, or `('XYZ', [0, 2, 5])`
- `.evaluate(sim, state) -> float`

### `GateFuser(max_qubits=3, max_window=None)`
- `__call__(circuit_or_instructions)` -> fused `Circuit` or list
- `.fusion_report(circuit) -> dict`: `{'original_count', 'fused_count', 'saved', 'blocks_by_size'}`

### `fuse(circuit, max_qubits=3, max_window=None) -> Circuit`

### `MPS`
- `.expectation_pauli_z/x/y(site) -> float`: single-site only
- `.expectation_single(op, site) -> complex`: single-site only, arbitrary 2x2 operator
- `.expectation_two(op, site_i) -> complex`: two-site operator on the
  adjacent pair `(site_i, site_i + 1)`
- `.expectation_two_site(op_i, op_j, site_i, site_j) -> complex`:
  two-site correlator `<op_i(site_i) op_j(site_j)>` for arbitrary,
  possibly non-adjacent sites
- `.expectation_multi_site(ops, sites) -> complex`: the general,
  correlated-system building block, `<op_0(sites[0]) op_1(sites[1]) ...>`
  for any number of single-site operators at arbitrary, possibly
  non-adjacent sites, computed in a single O(n * chi^3) sweep.
  `expectation_two_site` is a thin wrapper around this.
- `.apply_block_svd(start, block_tensor, k, chi=None, svd_threshold=1e-14) -> float`
  Generalizes `.apply_svd_truncation` (the two-site case) to a k-site
  block. This is the mechanism fused gates from `GateFuser` use to re-split
  back into the MPS.
- `.to_statevector() -> ndarray`  (n <= 20)
- `.bond_dimensions() -> list`, `.total_truncation_error() -> float`
- `.to(device) -> MPS`, `.copy() -> MPS`

> **Single-site vs. correlated systems:** `expectation_pauli_z/x/y` and
> `expectation_single` only ever act on one site and say nothing about
> correlations between qubits. To get expectation values of the
> correlated system, e.g. `<Z0 Z3>`, `<X1 Y4 Z7>`, use
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
mps-xtrap simulate    --circuit ghz --n 10 --chi 64
mps-xtrap extrapolate --n 8 --dt 0.2 0.1 0.05 0.025 --order 2 --chi 64
mps-xtrap benchmark   --circuit ising --n 8 --chi-start 8 --chi-levels 4
mps-xtrap info
```

> The installed console-script name is `mps-xtrap` (hyphen), as registered
> in `entry_points.txt`, not `mps_xtrap` (underscore, the importable
> Python package name). If you are running from source without installing
> the entry point, use `python -m mps_xtrap.cli ...` or `python
> mps_xtrap/cli.py ...` instead.

---

## Dependencies

- Python 3.8+
- NumPy >= 1.21
- CuPy (optional, for GPU)
