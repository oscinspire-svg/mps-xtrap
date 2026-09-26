#!/usr/bin/env python3
"""
mps_xtrap CLI — MPS quantum circuit simulator with overlapping-support gate
fusion and Trotter-step Richardson extrapolation.

Usage
-----
  python cli.py simulate    --circuit bell  --chi 64
  python cli.py extrapolate --n 8 --dt 0.2 0.1 0.05 0.025 --order 2
  python cli.py benchmark   --circuit ghz   --n 10
  python cli.py info
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import time
from mps_xtrap import Circuit, MPSSimulator, TrotterExtrapolator


# ---------------------------------------------------------------------------
# Built-in circuit library
# ---------------------------------------------------------------------------

def make_bell(n=2) -> Circuit:
    c = Circuit(max(2, n))
    c.h(0).cx(0, 1)
    return c

def make_ghz(n=6) -> Circuit:
    c = Circuit(n)
    c.h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c

def make_ising(n=8, steps=5, J=1.0, h=0.5, dt=0.05) -> Circuit:
    c = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            c.rx(2 * h * dt, q)
        for q in range(n - 1):
            c.zz(2 * J * dt, q, q + 1)
    return c

def make_ising_trotter(dt, n=8, J=1.0, h=0.5, t_total=1.0, order=2) -> Circuit:
    """
    Trotterized transverse-field Ising evolution at step size dt, holding
    the total evolution time t_total fixed (steps = round(t_total / dt)).

    order=1: first-order Lie-Trotter (field then interaction each step).
    order=2: second-order symmetric (Strang) splitting (half-field,
             interaction, half-field each step) — the default, and the
             usual choice for Trotter-step Richardson extrapolation.
    """
    steps = max(1, round(t_total / dt))
    c = Circuit(n)
    if order == 1:
        for _ in range(steps):
            for q in range(n):
                c.rx(2 * h * dt, q)
            for q in range(n - 1):
                c.zz(2 * J * dt, q, q + 1)
    else:
        for _ in range(steps):
            for q in range(n):
                c.rx(h * dt, q)
            for q in range(n - 1):
                c.zz(2 * J * dt, q, q + 1)
            for q in range(n):
                c.rx(h * dt, q)
    return c

def make_qft(n=6) -> Circuit:
    """Quantum Fourier Transform."""
    c = Circuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.cp(np.pi / 2 ** (j - i), i, j)
    for i in range(n // 2):
        c.swap(i, n - 1 - i)
    return c

def make_random(n=8, depth=10, seed=42) -> Circuit:
    np.random.seed(seed)
    c = Circuit(n)
    for _ in range(depth):
        for q in range(n):
            c.ry(np.random.uniform(0, np.pi), q)
        for q in range(n - 1):
            if np.random.random() < 0.5:
                c.cx(q, q + 1)
    return c

CIRCUITS = {
    'bell':   make_bell,
    'ghz':    make_ghz,
    'ising':  make_ising,
    'qft':    make_qft,
    'random': make_random,
}


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_simulate(args):
    print(f"\n── Simulation ──────────────────────────────────────────")
    circuit_fn = CIRCUITS.get(args.circuit)
    if circuit_fn is None:
        print(f"Unknown circuit '{args.circuit}'. Available: {list(CIRCUITS)}")
        return

    c = circuit_fn(n=args.n)
    print(f"Circuit: {c}")
    print(f"Bond dim: chi={args.chi}")

    t0 = time.time()
    sim = MPSSimulator(chi=args.chi)
    state = sim.run(c)
    elapsed = time.time() - t0

    print(f"\nResults (time: {elapsed:.2f}s):")
    for q in range(min(c.n, 10)):
        z = state.expectation_pauli_z(q)
        x = state.expectation_pauli_x(q)
        print(f"  q{q}: <Z>={z:+.6f}  <X>={x:+.6f}")

    print(f"\nBond dims: {state.bond_dimensions()}")
    print(f"Max bond dim reached: {state.max_bond_dim()} / {args.chi}")
    print(f"Total truncation error: {state.total_truncation_error():.2e}")
    print(f"State: {state}")


def cmd_extrapolate(args):
    print(f"\n── Trotter-Step Richardson Extrapolation ───────────────")

    n = args.n
    dt_values = sorted(args.dt, reverse=True)
    print(f"Qubits: {n}   Trotter order p: {args.order}   Total time: {args.t_total}")
    print(f"Step sizes (dt): {dt_values}")
    print()

    def circuit_fn(dt):
        return make_ising_trotter(
            dt, n=n, J=args.J, h=args.h, t_total=args.t_total, order=args.order,
        )

    observables = {f'Z{q}': ('Z', q) for q in range(min(n, args.obs_qubits))}
    extrapolator = TrotterExtrapolator(order=args.order, chi=args.chi, verbose=True)

    t0 = time.time()
    result = extrapolator.run(circuit_fn, dt_values, observables)
    elapsed = time.time() - t0

    print(f"\n{result.summary()}")
    print(f"Total runtime: {elapsed:.2f}s")


def cmd_benchmark(args):
    print(f"\n── Benchmark ───────────────────────────────────────────")
    circuit_fn = CIRCUITS.get(args.circuit)
    if circuit_fn is None:
        print(f"Unknown circuit '{args.circuit}'. Available: {list(CIRCUITS)}")
        return

    c = circuit_fn(n=args.n)
    print(f"Circuit: {c}")
    print(f"{'chi':>6}  {'Time(s)':>10}  {'<Z0>':>12}  {'TruncErr':>12}  {'MaxBond':>8}")
    print("-" * 60)

    chi_list = [args.chi_start * (2 ** i) for i in range(args.chi_levels)]

    for chi in chi_list:
        t0 = time.time()
        sim = MPSSimulator(chi=chi)
        state = sim.run(c)
        elapsed = time.time() - t0
        z0 = state.expectation_pauli_z(0)
        trunc = state.total_truncation_error()
        maxb = state.max_bond_dim()
        print(f"{chi:>6}  {elapsed:>10.3f}  {z0:>+12.8f}  {trunc:>12.2e}  {maxb:>8}")

    print(
        "\nFor extrapolating an observable to a limit (rather than just "
        "comparing chi values), see `mps_xtrap extrapolate`, which "
        "extrapolates Trotterized time evolution to the dt -> 0 limit."
    )


def cmd_info(args):
    print("""
mps_xtrap — MPS Quantum Circuit Simulator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABOUT
  Simulates quantum circuits using Matrix Product States (MPS), with two
  headline features:
    - Overlapping-support gate fusion: merges consecutive gates that
      share qubit support — not just gates on the identical qubit or
      qubit pair — into single combined gates, cutting the number of
      SVD truncations.
    - Trotter-step Richardson extrapolation: simulate a Trotterized
      time-evolution circuit at a few step sizes dt and extrapolate
      observables to the dt -> 0 continuum limit.

GATES SUPPORTED
  Single-qubit: I, X, Y, Z, H, S, T, Sdg, Tdg, Rx, Ry, Rz, P, U
  Two-qubit:    CNOT/CX, CZ, SWAP, iSWAP, XX, YY, ZZ, CRz, CP
  Fused gates:  automatically produced by GateFuser, up to max_qubits wide

BUILT-IN CIRCUITS
  bell    — 2-qubit Bell state
  ghz     — N-qubit GHZ state
  ising   — Transverse-field Ising Trotterized evolution (fixed dt)
  qft     — Quantum Fourier Transform
  random  — Random brick-layer circuit

FUSION THEORY
  A run of consecutive gates is fused whenever each new gate shares at
  least one qubit with the block built so far, up to max_qubits total
  qubits. Same-qubit chains and same-pair runs are the size-1 and size-2
  special cases; overlapping runs (e.g. CX(0,1) then CX(1,2)) fuse into
  wider gates that neither could reach alone. Fusion is exact — the fused
  gate is exactly the product of the originals — so it changes nothing
  but speed.

EXTRAPOLATION THEORY
  Trotterized time evolution with a p-th order product formula has error
  <O>(dt) ~ <O>(0) + c1*dt^p + c2*dt^(2p) + ..., a well-established result
  for product-formula simulation (Trotter 1959; Suzuki 1976). Richardson
  extrapolation on simulations at a few step sizes dt cancels successive
  powers of dt^p, extrapolating to the dt -> 0 (continuum) limit.
  Reliability diagnostics compare the assumed order p against a
  data-estimated order, and flag when MPS truncation error is not
  negligible next to the Trotter correction being extracted.

EXAMPLES
  python cli.py simulate    --circuit ghz --n 10 --chi 64
  python cli.py extrapolate --n 8 --dt 0.2 0.1 0.05 0.025 --order 2 --chi 64
  python cli.py benchmark   --circuit ising --n 8 --chi-start 8 --chi-levels 4
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='mps_xtrap — MPS Quantum Simulator with Richardson Extrapolation'
    )
    sub = parser.add_subparsers(dest='command')

    # simulate
    p_sim = sub.add_parser('simulate', help='Run a single simulation')
    p_sim.add_argument('--circuit', default='ghz', choices=list(CIRCUITS))
    p_sim.add_argument('--n', type=int, default=8, help='Number of qubits')
    p_sim.add_argument('--chi', type=int, default=64, help='Bond dimension')

    # extrapolate
    p_ext = sub.add_parser(
        'extrapolate',
        help='Trotter-step Richardson extrapolation of Ising time evolution',
    )
    p_ext.add_argument('--n', type=int, default=8, help='Number of qubits')
    p_ext.add_argument('--J', type=float, default=1.0, help='ZZ coupling strength')
    p_ext.add_argument('--h', type=float, default=0.5, help='Transverse field strength')
    p_ext.add_argument('--t-total', type=float, default=1.0,
                       help='Total physical evolution time (held fixed across dt)')
    p_ext.add_argument('--dt', type=float, nargs='+', default=[0.2, 0.1, 0.05, 0.025],
                       help='Trotter step sizes to simulate at (at least 2)')
    p_ext.add_argument('--order', type=int, default=2, choices=[1, 2],
                       help='Trotter product-formula order (1=Lie-Trotter, 2=Strang)')
    p_ext.add_argument('--chi', type=int, default=64, help='Bond dimension for every run')
    p_ext.add_argument('--obs-qubits', type=int, default=4,
                       help='Number of qubits to measure')

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Benchmark across chi values')
    p_bench.add_argument('--circuit', default='ising', choices=list(CIRCUITS))
    p_bench.add_argument('--n', type=int, default=8)
    p_bench.add_argument('--chi-start', type=int, default=8)
    p_bench.add_argument('--chi-levels', type=int, default=4)

    # info
    sub.add_parser('info', help='Show package info and supported gates')

    args = parser.parse_args()

    if args.command == 'simulate':
        cmd_simulate(args)
    elif args.command == 'extrapolate':
        cmd_extrapolate(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
