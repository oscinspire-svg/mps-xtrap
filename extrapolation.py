"""
Trotter-Step Richardson Extrapolation for MPS quantum circuit simulation.

Theory
------
Simulating time evolution under a Hamiltonian ``H = sum_j H_j`` requires
approximating ``e^{-iHt}`` by a product formula ("Trotterization") that
splits the evolution into a sequence of exponentials of the individual
terms, applied step by step:

    e^{-iHt}  ~  [ S(dt) ]^{t/dt}

For a p-th order product formula ``S`` — p=1 for first-order Lie-Trotter
splitting, p=2 for the symmetric (Strang) splitting, p=4, 6, ... for
higher-order Suzuki formulas — the error this approximation introduces
into an expectation value ``<O>`` computed from the Trotterized circuit
has the well-established form:

    <O>(dt)  ~  <O>(0) + c_1 dt^p + c_2 dt^{2p} + ...

where ``<O>(0)`` is the exact, continuous-time (dt -> 0) value. This is
the standard error expansion for product-formula time evolution (Trotter
1959; Suzuki 1976; see Childs, Su, Tran, Wiebe & Zhu (2021), "Theory of
Trotter Error", for rigorous bounds).

Because the leading error term is a *known* power of ``dt`` — fixed by
which splitting order was used to build the circuit, not something that
has to be inferred from simulation data — Richardson extrapolation
applies directly: simulate the same physical evolution (fixed total time
``T``) at several step sizes ``dt_0 > dt_1 > ... > dt_{k-1}`` (using more,
smaller Trotter steps as ``dt`` shrinks) and extrapolate the sequence of
results to the ``dt -> 0`` limit.

Richardson's recursive formula, in the general form valid for *any*
sequence of step sizes (not just a fixed geometric ratio):

    T[i, 0] = <O>(dt_i)
    T[i, m] = T[i, m-1] + ( T[i, m-1] - T[i-1, m-1] ) / ( (dt_{i-m} / dt_i)^p - 1 )

for ``m = 1, ..., i``. This is the same recursive construction used in
Romberg integration, generalized to an arbitrary (not necessarily
geometric) sequence of step sizes; each level cancels one more power of
``dt^p`` from the error expansion, so ``T[k-1, k-1]`` — built from all
``k`` simulated step sizes — has error formally of order ``O(dt^{k*p})``.

Reliability diagnostics:
  - Monotone-convergence check on the raw values as dt shrinks
  - Richardson-correction decay check across extrapolation levels
  - Uncertainty estimate from the size of the last correction
  - Order-consistency check: the assumed order p is compared against an
    order estimated directly from the data (log-log fit)
  - MPS truncation-error check: flags when bond-dimension truncation error
    is not safely smaller than the Trotter correction being extracted,
    since in that regime the "dt -> 0" extrapolation would really be
    extrapolating MPS truncation noise, not Trotter error

References
----------
- Trotter (1959), Suzuki (1976) for product ("Trotter") formulas
- Childs, Su, Tran, Wiebe, Zhu (2021), "Theory of Trotter Error"
- Richardson (1911), Romberg (1955) for the extrapolation framework
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable, Sequence
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtrapolationResult:
    """
    Result of Trotter-step Richardson extrapolation for a single observable.

    Attributes
    ----------
    dt_values : list of float
        Trotter step sizes used, sorted largest to smallest.
    raw_values : list of float
        Expectation values simulated at each step size (same order as
        ``dt_values``).
    extrapolated : float
        Best estimate of the dt -> 0 (continuum) value.
    uncertainty : float
        Estimated uncertainty of the extrapolated value.
    table : ndarray
        Full Richardson tableau (table[i, m] as defined in the module
        docstring; NaN where undefined).
    convergence_ratios : list of float
        Ratios of successive Richardson corrections along the final row
        (should shrink for a reliable extrapolation).
    is_reliable : bool
        Whether the extrapolation passes reliability diagnostics.
    reliability_notes : list of str
        Explanations of any reliability concerns.
    order : int
        The assumed Trotter product-formula order p used for extrapolation.
    estimated_order : float or None
        Order inferred directly from the raw data via log-log regression,
        reported as a consistency check against ``order``.
    max_truncation_error : float or None
        Largest per-run MPS truncation error observed across the dt sweep,
        if truncation-error tracking was available.
    """
    dt_values: List[float]
    raw_values: List[float]
    extrapolated: float
    uncertainty: float
    table: NDArray
    convergence_ratios: List[float]
    is_reliable: bool
    reliability_notes: List[str]
    order: int
    estimated_order: Optional[float] = None
    max_truncation_error: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "Trotter-Step Richardson Extrapolation Result",
            "=" * 62,
            f"Step sizes (dt):  {self.dt_values}",
            f"Raw values:       {[f'{v:.8f}' for v in self.raw_values]}",
            f"Extrapolated:     {self.extrapolated:.8f} +/- {self.uncertainty:.2e}",
            f"Assumed order p:  {self.order}",
        ]
        if self.estimated_order is not None:
            lines.append(f"Data-estimated p: {self.estimated_order:.2f}")
        if self.max_truncation_error is not None:
            lines.append(f"Max MPS trunc err: {self.max_truncation_error:.2e}")
        lines.append(f"Reliable:         {self.is_reliable}")
        if self.reliability_notes:
            lines.append("Notes:")
            for note in self.reliability_notes:
                lines.append(f"  - {note}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def improvement_factor(self) -> float:
        """How much the extrapolation improved over the finest raw value."""
        finest_raw = self.raw_values[-1]
        if abs(self.extrapolated - finest_raw) < 1e-15:
            return 1.0
        return abs(self.raw_values[0] - self.raw_values[-1]) / \
               max(abs(self.extrapolated - finest_raw), 1e-15)


@dataclass
class MultiObservableResult:
    """Results for multiple observables extrapolated over the same dt sweep."""
    observables: Dict[str, ExtrapolationResult]
    dt_values: List[float]

    def summary(self) -> str:
        lines = ["Multi-Observable Trotter Richardson Extrapolation", "=" * 55]
        lines.append(f"Step sizes (dt): {self.dt_values}")
        for name, res in self.observables.items():
            flag = "OK " if res.is_reliable else "!! "
            lines.append(
                f"  {flag}{name:20s}: {res.extrapolated:.8f} +/- {res.uncertainty:.2e}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class TrotterSweepConfig:
    """
    Declarative step-size schedule for a Trotter sweep.

    Instead of typing out an explicit ``dt_values`` list, describe the
    sweep by its coarsest step size, how much finer each successive level
    gets, and how many levels to run:

        dt_values[i] = base_dt / (refinement_ratio ** i),  i = 0 .. n_levels-1

    So ``base_dt`` is the *coarsest* (largest) step size, and each level
    refines to a step ``refinement_ratio`` times *smaller* than the one
    before it.

    Parameters
    ----------
    base_dt : float
        Coarsest (largest) Trotter step size — level 0 of the sweep.
    refinement_ratio : float
        How much finer each successive level's step size gets. Must be
        > 1. Must be supplied explicitly — there is no default, since
        silently assuming 2.0 (halving) can mask a mismatch with the
        Trotter/Suzuki order actually used to build the circuit.
    n_levels : int
        Total number of step sizes / simulation runs in the sweep. Must
        be >= 2 (Richardson extrapolation needs at least two points).
        Must be supplied explicitly — there is no default, since a
        silently-assumed sweep length can hide an under-resolved
        extrapolation.

    Example
    -------
    >>> config = TrotterSweepConfig(base_dt=0.2, refinement_ratio=2.0, n_levels=4)
    >>> config.dt_values
    [0.2, 0.1, 0.05, 0.025]

    Calling the config runs a circuit once per level, applying that
    level's step size each time — this is the "collect" step, producing
    a :class:`TrotterSweepResult` that :meth:`TrotterExtrapolator.extrapolate`
    then consumes:

    >>> sweep_result = config(trotter_circuit, chi=64, observables={'Z0': ('Z', 0)})
    >>> result = TrotterExtrapolator(order=2).extrapolate(sweep_result)
    """
    base_dt: float
    refinement_ratio: float
    n_levels: int

    def __post_init__(self):
        if self.base_dt <= 0:
            raise ValueError("base_dt must be positive.")
        if self.refinement_ratio <= 1.0:
            raise ValueError(
                "refinement_ratio must be > 1 (each level must refine to a "
                "smaller step size than the one before it)."
            )
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2 — Richardson extrapolation needs at least 2 points.")

    @property
    def dt_values(self) -> List[float]:
        """The step size at each level, coarsest (level 0) to finest (level n_levels-1)."""
        return [self.base_dt / (self.refinement_ratio ** i) for i in range(self.n_levels)]

    def __call__(
        self,
        circuit_fn: Callable[[float], "object"],
        chi: int = 64,
        device: str = 'cpu',
        observables: Optional[Dict[str, Tuple]] = None,
        measure: Optional[Dict[str, Tuple[int, str]]] = None,
        verbose: bool = True,
    ) -> "TrotterSweepResult":
        """
        Run ``circuit_fn(dt)`` once per level of this sweep — ``n_levels``
        runs total — applying that level's step size each time, and
        collect the raw per-level results. This is the "collect" phase:
        it does not extrapolate. Pass the returned :class:`TrotterSweepResult`
        to :meth:`TrotterExtrapolator.extrapolate` to get the ``dt -> 0``
        estimate.

        Parameters
        ----------
        circuit_fn : callable
            Given a step size ``dt``, returns the Trotterized ``Circuit``
            for that step size. Must hold the total physical evolution
            time fixed across calls (e.g. ``steps = round(t_total / dt)``)
            so every level targets the same underlying evolution.
        chi, device : bond dimension and backend, used for every level.
        observables, measure : same specs as :meth:`TrotterExtrapolator.run`.
        verbose : bool
            Print progress for each level.

        Returns
        -------
        TrotterSweepResult
        """
        from .simulator import MPSSimulator
        from .measurement import MeasurementEngine

        observables = observables or {}
        measure = measure or {}
        dt_list = self.dt_values

        all_names = list(observables) + list(measure)
        values_dict: Dict[str, List[float]] = {name: [] for name in all_names}
        truncation_errors: List[float] = []

        for level, dt in enumerate(dt_list):
            if verbose:
                print(
                    f"  [level {level + 1}/{len(dt_list)}] dt={dt:g}...",
                    end=" ", flush=True,
                )

            circuit = circuit_fn(dt)
            sim = MPSSimulator(chi=chi, device=device)
            state = sim.run(circuit)
            trunc_err = state.total_truncation_error()
            truncation_errors.append(trunc_err)

            for name, spec in observables.items():
                obs, rest = spec[0], spec[1:]
                if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
                    sites = list(rest[0])
                    value = sim.expectation_correlator(state, obs, sites)
                else:
                    site = rest[0]
                    site2 = rest[1] if len(rest) > 1 else None
                    value = sim.expectation_value(state, obs, site, site2)
                values_dict[name].append(value)

            if measure:
                meng = MeasurementEngine(state)
                for name, (site, outcome_key) in measure.items():
                    mres = meng.measure_qubit(site, collapse=False)
                    if outcome_key == 'prob0':
                        values_dict[name].append(mres.prob0)
                    elif outcome_key == 'prob1':
                        values_dict[name].append(mres.prob1)
                    else:
                        raise ValueError(
                            f"Unknown outcome_key {outcome_key!r} for '{name}'. "
                            "Use 'prob0' or 'prob1'."
                        )

            if verbose:
                print(f"done (trunc_err={trunc_err:.2e})")

        return TrotterSweepResult(
            config=self,
            dt_values=dt_list,
            values=values_dict,
            truncation_errors=truncation_errors,
        )


@dataclass
class TrotterSweepResult:
    """
    Raw (not yet extrapolated) results of running a Trotterized circuit
    once per step size in a :class:`TrotterSweepConfig`.

    Attributes
    ----------
    config : TrotterSweepConfig
        The sweep schedule that produced this result.
    dt_values : list of float
        Step size used at each level (coarsest to finest).
    values : dict
        Mapping of observable/measurement label to a list of raw values,
        one per level, in the same order as ``dt_values``.
    truncation_errors : list of float
        MPS truncation error observed at each level.
    """
    config: TrotterSweepConfig
    dt_values: List[float]
    values: Dict[str, List[float]]
    truncation_errors: List[float]


# ---------------------------------------------------------------------------
# Core extrapolation engine (internal)
# ---------------------------------------------------------------------------

class _RichardsonExtrapolator:
    """
    Internal Richardson extrapolation engine for Trotter-step sequences.

    Not part of the public API — use :class:`TrotterExtrapolator`, which
    drives simulation and calls into this.

    Parameters
    ----------
    order : int
        Known order p of the Trotter/Suzuki product formula used to build
        the circuits being extrapolated (1, 2, 4, 6, ...).
    """

    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order must be >= 1.")
        self.order = order

    @staticmethod
    def _estimate_order(dt_values: List[float], values: List[float]) -> Optional[float]:
        """
        Estimate the Trotter error order p from the data via a log-log
        regression of successive-difference magnitude vs dt. Purely a
        diagnostic; the extrapolation itself always uses the known,
        physically-motivated ``order``.
        """
        if len(values) < 3:
            return None

        diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        dts = [dt_values[i] for i in range(len(diffs))]
        keep = [(d, dt) for d, dt in zip(diffs, dts) if d > 1e-15]
        if len(keep) < 2:
            return None

        diffs_k, dts_k = zip(*keep)
        coeffs = np.polyfit(np.log(dts_k), np.log(diffs_k), 1)
        return float(coeffs[0])

    def _run(
        self,
        dt_values: List[float],
        values: List[float],
        truncation_errors: Optional[List[float]] = None,
    ) -> ExtrapolationResult:
        """
        Apply Trotter-step Richardson extrapolation to a sequence of
        (dt, value) pairs.

        Parameters
        ----------
        dt_values : list of float
            Step sizes, any order on input; sorted descending internally.
        values : list of float
            Observed expectation values, matching ``dt_values`` before
            sorting.
        truncation_errors : list of float, optional
            Per-run MPS truncation error, matching ``dt_values`` before
            sorting, used for the truncation-error reliability check.
        """
        if len(dt_values) != len(values):
            raise ValueError("dt_values and values must have the same length.")
        if len(dt_values) < 2:
            raise ValueError("Need at least 2 step sizes to extrapolate.")
        if len(set(dt_values)) != len(dt_values):
            raise ValueError("dt_values must be distinct.")

        # Sort descending (largest step / coarsest first), matching the
        # convention h_0 > h_1 > ... used in Romberg-style tableaus.
        order_idx = sorted(range(len(dt_values)), key=lambda i: -dt_values[i])
        dt = [dt_values[i] for i in order_idx]
        vals = [values[i] for i in order_idx]
        trunc = [truncation_errors[i] for i in order_idx] if truncation_errors else None

        p = self.order
        k = len(dt)

        table = np.full((k, k), np.nan)
        table[:, 0] = vals

        for i in range(1, k):
            for m in range(1, i + 1):
                ratio = (dt[i - m] / dt[i]) ** p
                denom = ratio - 1.0
                if abs(denom) < 1e-300:
                    table[i, m] = table[i, m - 1]
                else:
                    table[i, m] = table[i, m - 1] + \
                        (table[i, m - 1] - table[i - 1, m - 1]) / denom

        extrapolated = table[k - 1, k - 1]

        # Uncertainty: magnitude of the last correction applied, i.e. how
        # much the final level changed the previous-level estimate. Falls
        # back to the raw difference between the two finest points if only
        # two step sizes were given.
        if k >= 3:
            uncertainty = abs(table[k - 1, k - 1] - table[k - 1, k - 2])
        else:
            uncertainty = abs(vals[-1] - vals[-2])

        # Corrections along the most-refined diagonal entry's row, i.e.
        # how much each successive level changed the k-1'th row's estimate.
        corrections = [
            abs(table[k - 1, m] - table[k - 1, m - 1])
            for m in range(1, k)
            if not np.isnan(table[k - 1, m]) and not np.isnan(table[k - 1, m - 1])
        ]
        conv_ratios = [
            corrections[i] / corrections[i + 1]
            for i in range(len(corrections) - 1)
            if corrections[i + 1] > 1e-15
        ]

        estimated_order = self._estimate_order(dt, vals)
        max_trunc = max(trunc) if trunc else None

        notes = []
        is_reliable = True

        # Check 1: raw values roughly monotone as dt shrinks.
        diffs = [vals[i + 1] - vals[i] for i in range(k - 1)]
        if not (all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)):
            notes.append(
                "Raw values are not monotonically converging as dt shrinks."
            )
            is_reliable = False

        # Check 2: Richardson corrections should shrink from level to level.
        if len(corrections) >= 2 and corrections[-1] > corrections[-2] * 1.5:
            notes.append(
                "Richardson corrections are not decreasing — extrapolation "
                "may be unreliable. Consider adding a smaller dt."
            )
            is_reliable = False

        # Check 3: uncertainty relative to signal size.
        signal = abs(extrapolated)
        if signal > 1e-10 and uncertainty / signal > 0.5:
            notes.append(
                f"Uncertainty ({uncertainty:.2e}) is large relative to the "
                f"signal ({signal:.2e}). Consider adding more/smaller dt points."
            )
            is_reliable = False

        # Check 4: assumed order p is consistent with what the data shows.
        if estimated_order is not None:
            order_tolerance = max(0.75, 0.5 * p)
            if abs(estimated_order - p) > order_tolerance:
                notes.append(
                    f"Assumed order p={p} disagrees with the data-estimated "
                    f"order ({estimated_order:.2f}). Double check which "
                    "Trotter/Suzuki formula the circuit actually implements."
                )
                is_reliable = False

        # Check 5: MPS truncation error should be well below the Trotter
        # correction being extracted, or the extrapolation is really
        # chasing bond-dimension noise instead of Trotter error.
        if max_trunc is not None and corrections:
            finest_correction = corrections[-1]
            if max_trunc > 0.1 * max(finest_correction, 1e-15):
                notes.append(
                    f"Max MPS truncation error ({max_trunc:.2e}) is not "
                    f"negligible next to the finest Richardson correction "
                    f"({finest_correction:.2e}). Increase chi so the dt -> 0 "
                    "extrapolation isn't contaminated by bond-dimension error."
                )
                is_reliable = False

        if not notes:
            notes.append("All reliability checks passed.")

        return ExtrapolationResult(
            dt_values=dt,
            raw_values=vals,
            extrapolated=float(np.real(extrapolated)),
            uncertainty=float(uncertainty),
            table=table,
            convergence_ratios=conv_ratios,
            is_reliable=is_reliable,
            reliability_notes=notes,
            order=p,
            estimated_order=estimated_order,
            max_truncation_error=max_trunc,
        )

    def _run_multi(
        self,
        dt_values: List[float],
        values_dict: Dict[str, List[float]],
        truncation_errors: Optional[List[float]] = None,
    ) -> MultiObservableResult:
        """Extrapolate multiple observables over the same dt sweep."""
        results = {}
        for obs_name, vals in values_dict.items():
            try:
                results[obs_name] = self._run(dt_values, vals, truncation_errors)
            except Exception as e:
                logger.warning(f"Extrapolation failed for {obs_name}: {e}")
        return MultiObservableResult(observables=results, dt_values=list(dt_values))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class TrotterExtrapolator:
    """
    The public entry point for Trotter-step Richardson extrapolation.

    Simulates a Trotterized circuit at several step sizes ``dt`` — holding
    the total physical evolution time fixed by using more, smaller steps
    as ``dt`` shrinks — and extrapolates observables to the ``dt -> 0``
    (continuum) limit using Richardson extrapolation, exploiting the known
    ``O(dt^p)`` error order of whichever product formula built the circuit.

    Parameters
    ----------
    order : int
        Order p of the Trotter/Suzuki product formula used by
        ``circuit_fn`` (1 = first-order Lie-Trotter, 2 = symmetric/Strang
        splitting, 4, 6, ... for higher-order Suzuki formulas). Default 2,
        the most common choice.
    chi : int
        Bond dimension used for every simulation in the sweep. Because
        this module targets the ``dt -> 0`` error specifically, ``chi``
        should be large enough that MPS truncation error stays negligible
        across the whole sweep — see ``ExtrapolationResult.max_truncation_error``
        and the reliability notes, which flag it if not.
    device : str
        'cpu' (default) or 'cuda'.
    verbose : bool
        Print progress during simulation.

    Example
    -------
        def trotter_circuit(dt, n=8, J=1.0, h=0.5, t_total=1.0):
            '''Second-order (Strang) Trotterized Ising evolution.'''
            steps = round(t_total / dt)
            c = Circuit(n)
            for _ in range(steps):
                for q in range(n):
                    c.rx(h * dt, q)              # half-step field
                for q in range(n - 1):
                    c.zz(2 * J * dt, q, q + 1)    # full-step interaction
                for q in range(n):
                    c.rx(h * dt, q)              # half-step field
            return c

        extrapolator = TrotterExtrapolator(order=2, chi=64)
        result = extrapolator.run(
            trotter_circuit,
            dt_values=[0.2, 0.1, 0.05, 0.025],
            observables={'Z0': ('Z', 0)},
        )
        print(result.summary())
    """

    def __init__(
        self,
        order: int = 2,
        chi: int = 64,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        self.order = order
        self.chi = chi
        self.device = device
        self.verbose = verbose
        self._extrapolator = _RichardsonExtrapolator(order=order)

    def run(
        self,
        circuit_fn: Callable[[float], "object"],
        dt_values: Sequence[float],
        observables: Optional[Dict[str, Tuple]] = None,
        measure: Optional[Dict[str, Tuple[int, str]]] = None,
    ) -> MultiObservableResult:
        """
        Simulate ``circuit_fn(dt)`` at each step size and extrapolate.

        Parameters
        ----------
        circuit_fn : callable
            Given a step size ``dt``, returns the ``Circuit`` implementing
            Trotterized evolution at that step size. It is the caller's
            responsibility to keep the total physical evolution time fixed
            across calls (e.g. by using ``round(t_total / dt)`` steps) —
            the extrapolation is only meaningful if every simulated circuit
            approximates the *same* target evolution.
        dt_values : sequence of float
            Step sizes to simulate at (at least 2, all distinct; any
            order on input). Smaller values give a more accurate but more
            expensive simulation; the spread should be wide enough for
            Richardson extrapolation to have real error orders to cancel.
        observables : dict, optional
            Mapping of label to an observable spec, one of:
              - ``(observable_type, site)`` for a single-site observable,
                e.g. ``{'Z0': ('Z', 0)}``
              - ``(observable_type, site, site2)`` for a two-qubit
                correlator, e.g. ``{'ZZ_03': ('ZZ', 0, 3)}``
              - ``(observable_type, [site0, site1, ...])`` for a
                correlator over any number of sites, e.g.
                ``{'ZXZ_025': ('ZXZ', [0, 2, 5])}``
        measure : dict, optional
            Mapping of label to ``(site, outcome_key)`` for Born-rule
            probabilities, ``outcome_key`` one of ``'prob0'``/``'prob1'``.

        Returns
        -------
        MultiObservableResult
        """
        from .simulator import MPSSimulator
        from .measurement import MeasurementEngine

        observables = observables or {}
        measure = measure or {}
        dt_list = list(dt_values)

        all_names = list(observables) + list(measure)
        values_dict: Dict[str, List[float]] = {name: [] for name in all_names}
        truncation_errors: List[float] = []

        for dt in dt_list:
            if self.verbose:
                print(f"  Simulating dt={dt:g}...", end=" ", flush=True)

            circuit = circuit_fn(dt)
            sim = MPSSimulator(chi=self.chi, device=self.device)
            state = sim.run(circuit)
            truncation_errors.append(state.total_truncation_error())

            for name, spec in observables.items():
                obs, rest = spec[0], spec[1:]
                if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
                    sites = list(rest[0])
                    value = sim.expectation_correlator(state, obs, sites)
                else:
                    site = rest[0]
                    site2 = rest[1] if len(rest) > 1 else None
                    value = sim.expectation_value(state, obs, site, site2)
                values_dict[name].append(value)

            if measure:
                meng = MeasurementEngine(state)
                for name, (site, outcome_key) in measure.items():
                    mres = meng.measure_qubit(site, collapse=False)
                    if outcome_key == 'prob0':
                        values_dict[name].append(mres.prob0)
                    elif outcome_key == 'prob1':
                        values_dict[name].append(mres.prob1)
                    else:
                        raise ValueError(
                            f"Unknown outcome_key {outcome_key!r} for '{name}'. "
                            "Use 'prob0' or 'prob1'."
                        )

            if self.verbose:
                print(f"done (trunc_err={state.total_truncation_error():.2e})")

        return self._extrapolator._run_multi(dt_list, values_dict, truncation_errors)

    def extrapolate(self, sweep_result: TrotterSweepResult) -> MultiObservableResult:
        """
        Richardson-extrapolate a :class:`TrotterSweepResult` already
        collected by calling a :class:`TrotterSweepConfig`.

        This is the "extrapolate" half of the collect/extrapolate split:
        the (expensive) simulation work already happened when the config
        was called; this just runs the (cheap) Richardson tableau on the
        stored per-level data. You can call this multiple times on the
        same collected sweep — e.g. after changing ``self.order`` to see
        how sensitive the result is — without re-simulating anything.

        Parameters
        ----------
        sweep_result : TrotterSweepResult
            Output of calling a :class:`TrotterSweepConfig`.

        Returns
        -------
        MultiObservableResult
        """
        return self._extrapolator._run_multi(
            sweep_result.dt_values, sweep_result.values, sweep_result.truncation_errors,
        )

    def run_sweep(
        self,
        circuit_fn: Callable[[float], "object"],
        config: TrotterSweepConfig,
        observables: Optional[Dict[str, Tuple]] = None,
        measure: Optional[Dict[str, Tuple[int, str]]] = None,
    ) -> MultiObservableResult:
        """
        One-shot convenience: collect a sweep from ``config`` and
        extrapolate it, using this extrapolator's ``chi``/``device``/
        ``verbose`` settings. Equivalent to::

            sweep_result = config(circuit_fn, chi=self.chi, device=self.device,
                                   observables=observables, measure=measure,
                                   verbose=self.verbose)
            result = self.extrapolate(sweep_result)

        Prefer calling ``config(...)`` and :meth:`extrapolate` separately
        if you want to keep the raw sweep data around (e.g. to try a
        different assumed ``order`` without re-simulating).
        """
        sweep_result = config(
            circuit_fn,
            chi=self.chi,
            device=self.device,
            observables=observables,
            measure=measure,
            verbose=self.verbose,
        )
        return self.extrapolate(sweep_result)


def trotter_extrapolate(
    circuit_fn: Callable[[float], "object"],
    dt_values: Sequence[float],
    order: int = 2,
    chi: int = 64,
    device: str = 'cpu',
    observables: Optional[Dict[str, Tuple]] = None,
    measure: Optional[Dict[str, Tuple[int, str]]] = None,
    verbose: bool = True,
) -> MultiObservableResult:
    """
    Convenience function wrapping :class:`TrotterExtrapolator` for one-off use.

    Example
    -------
    >>> result = trotter_extrapolate(
    ...     trotter_circuit, dt_values=[0.2, 0.1, 0.05],
    ...     order=2, chi=64, observables={'Z0': ('Z', 0)},
    ... )
    >>> print(result.summary())
    """
    return TrotterExtrapolator(order=order, chi=chi, device=device, verbose=verbose).run(
        circuit_fn, dt_values, observables=observables, measure=measure,
    )


__all__ = [
    "TrotterExtrapolator",
    "trotter_extrapolate",
    "TrotterSweepConfig",
    "TrotterSweepResult",
    "ExtrapolationResult",
    "MultiObservableResult",
]
