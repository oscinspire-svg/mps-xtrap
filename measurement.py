"""
Qubit measurement via projector operators for MPS states.

Theory
------
A projective measurement on qubit site s for outcome k ∈ {0, 1} is defined
by the projector:

    P_k = |k><k|    (a 2×2 matrix with a single 1 on the diagonal)

    P_0 = [[1, 0],   (projects onto |0>)
           [0, 0]]

    P_1 = [[0, 0],   (projects onto |1>)
           [0, 1]]

The Born-rule probability of outcome k on site s is:

    prob(k, s) = <ψ| P_k^(s) |ψ>

which is just a single-site expectation value of the projector operator.
This is computed efficiently using the MPS canonicalization machinery.

Post-measurement state
----------------------
After measuring qubit s and observing outcome k, the (unnormalized)
post-measurement state is:

    |ψ'> = P_k^(s) |ψ>

To obtain the normalized post-measurement MPS, we:
  1. Apply P_k (2×2 matrix) to site s as a single-qubit gate.
  2. Divide the tensor at site s by sqrt(prob(k, s)) to renormalize.

Multi-qubit measurements
------------------------
A bitstring measurement on qubits (s0, s1, ..., sm) computes the joint
probability:

    prob(b0, b1, ..., bm) = <ψ| P_b0^(s0) ⊗ P_b1^(s1) ⊗ ... |ψ>

This is computed as a chain of projections followed by norm-squared:

    p = ||P_bm^(sm) ··· P_b1^(s1) P_b0^(s0) |ψ>||²

A full measurement distribution (all 2^n bitstrings) is obtained by
computing the statevector |sv|² — but only available for small n.

For large n, sample_counts uses a sequential single-qubit measurement
strategy that is linear in the number of qubits.

Trotter-step extrapolation of probabilities
--------------------------------------------
Probabilities are expectation values of projector operators, so they
obey the same O(dt^p) Trotter error law as ordinary observables when
computed from a Trotterized time-evolution circuit:

    prob(k, s; dt) ~= prob(k, s; 0) + c*dt^p + ...

TrotterExtrapolator is extended via the 'measure' dict argument to sweep
over Trotter step sizes and extrapolate each probability to dt -> 0
alongside ordinary expectation values.

    result = extrapolator.run(
        trotter_circuit,
        observables={'Z0': ('Z', 0)},
        measure={'P0_q1': (1, 'prob0'), 'P1_q1': (1, 'prob1')},
    )

The extrapolated probabilities appear under result.observables alongside
ordinary expectation values.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
import logging

from .mps import MPS, _to_backend, _to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projector operators
# ---------------------------------------------------------------------------

def projector(outcome: int) -> NDArray:
    """
    Return the projector operator for a single-qubit measurement outcome.

    Parameters
    ----------
    outcome : int
        0 → P_0 = |0><0| = diag(1, 0)
        1 → P_1 = |1><1| = diag(0, 1)

    Returns
    -------
    NDArray of shape (2, 2)
    """
    if outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
    P = np.zeros((2, 2), dtype=complex)
    P[outcome, outcome] = 1.0
    return P


P0 = projector(0)   # |0><0|
P1 = projector(1)   # |1><1|


# ---------------------------------------------------------------------------
# Measurement result data classes
# ---------------------------------------------------------------------------

@dataclass
class SingleMeasurementResult:
    """
    Result of measuring a single qubit.

    Attributes
    ----------
    site : int
        Qubit index measured.
    prob0 : float
        Born-rule probability of outcome |0>.
    prob1 : float
        Born-rule probability of outcome |1>.
    post_state_0 : MPS or None
        Normalized post-measurement MPS after outcome 0, if requested.
    post_state_1 : MPS or None
        Normalized post-measurement MPS after outcome 1, if requested.
    """
    site: int
    prob0: float
    prob1: float
    post_state_0: Optional[MPS] = field(default=None, repr=False)
    post_state_1: Optional[MPS] = field(default=None, repr=False)

    def __post_init__(self):
        # Clamp to [0, 1] to guard against tiny floating-point overshoots
        self.prob0 = float(np.clip(self.prob0, 0.0, 1.0))
        self.prob1 = float(np.clip(self.prob1, 0.0, 1.0))

    def summary(self) -> str:
        return (
            f"Measurement(site={self.site}): "
            f"P(0)={self.prob0:.6f}, P(1)={self.prob1:.6f}, "
            f"sum={self.prob0+self.prob1:.8f}"
        )

    def most_likely(self) -> int:
        """Return the most likely outcome (0 or 1)."""
        return 0 if self.prob0 >= self.prob1 else 1


@dataclass
class MultiQubitMeasurementResult:
    """
    Result of measuring multiple qubits.

    Attributes
    ----------
    sites : list of int
        Qubit indices measured (in order).
    probabilities : dict
        Maps bitstring tuple → float probability.
        e.g. {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}
    marginals : dict
        Maps site index → SingleMeasurementResult (marginal over other sites).
    """
    sites: List[int]
    probabilities: Dict[Tuple[int, ...], float]
    marginals: Dict[int, SingleMeasurementResult]

    def summary(self) -> str:
        lines = [
            f"Multi-qubit measurement on sites {self.sites}",
            "-" * 44,
        ]
        # Sort by bitstring
        for bits, prob in sorted(self.probabilities.items()):
            bs = ''.join(map(str, bits))
            bar = '█' * int(prob * 20)
            lines.append(f"  |{bs}⟩ : {prob:.6f}  {bar}")
        lines.append(f"  total = {sum(self.probabilities.values()):.8f}")
        return '\n'.join(lines)

    def most_likely_bitstring(self) -> Tuple[int, ...]:
        """Return the bitstring with the highest probability."""
        return max(self.probabilities, key=lambda k: self.probabilities[k])


@dataclass
class SampledCounts:
    """
    Simulated measurement counts from repeated shots.

    Attributes
    ----------
    sites : list of int
        Qubit indices measured.
    counts : dict
        Maps bitstring string → int count.
    shots : int
        Total number of shots.
    """
    sites: List[int]
    counts: Dict[str, int]
    shots: int

    def probabilities(self) -> Dict[str, float]:
        """Normalize counts to empirical probabilities."""
        return {k: v / self.shots for k, v in self.counts.items()}

    def summary(self) -> str:
        lines = [f"SampledCounts(sites={self.sites}, shots={self.shots})", "-" * 44]
        for bs, cnt in sorted(self.counts.items(), key=lambda x: -x[1]):
            bar = '█' * int(cnt * 20 // self.shots)
            lines.append(f"  |{bs}⟩ : {cnt:5d}  {bar}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Core measurement engine
# ---------------------------------------------------------------------------

class MeasurementEngine:
    """
    Compute projective measurement probabilities and post-measurement states
    from an MPS, using projector operators.

    All computations are performed on the device of the input MPS
    (CPU or GPU via CuPy).

    Parameters
    ----------
    state : MPS
        The quantum state to measure. Not modified.
    """

    def __init__(self, state: MPS):
        self.state = state

    # ------------------------------------------------------------------
    # Single-qubit measurement
    # ------------------------------------------------------------------

    def measure_qubit(
        self,
        site: int,
        collapse: bool = False,
    ) -> SingleMeasurementResult:
        """
        Measure a single qubit by computing projector expectation values.

        Uses the identity:
            P(outcome=k) = <ψ| P_k |ψ>    where  P_k = |k><k|

        Parameters
        ----------
        site : int
            Qubit index to measure.
        collapse : bool
            If True, also compute the normalized post-measurement MPS for
            each outcome and include it in the result.

        Returns
        -------
        SingleMeasurementResult
        """
        if site < 0 or site >= self.state.n:
            raise ValueError(f"site {site} out of range [0, {self.state.n-1}]")

        p0 = self._prob_outcome(site, 0)
        p1 = self._prob_outcome(site, 1)

        post0 = self._collapse_to(site, 0, p0) if collapse and p0 > 1e-15 else None
        post1 = self._collapse_to(site, 1, p1) if collapse and p1 > 1e-15 else None

        return SingleMeasurementResult(
            site=site,
            prob0=p0,
            prob1=p1,
            post_state_0=post0,
            post_state_1=post1,
        )

    def _prob_outcome(self, site: int, outcome: int) -> float:
        """
        Compute P(outcome) = <ψ| P_outcome |ψ> at site.

        P_outcome is a (2,2) projector. This equals expectation_single(P_outcome, site).
        """
        P = projector(outcome)
        val = self.state.expectation_single(P, site)
        return float(np.real(_to_numpy(np.array(val))))

    def _collapse_to(self, site: int, outcome: int, prob: float) -> MPS:
        """
        Apply projector P_outcome to site, renormalize, return new MPS.

        The projector P_outcome is applied as a (2,2) gate to site s of a
        copy of the MPS, then the tensor is rescaled by 1/sqrt(prob).
        """
        new_state = self.state.copy()
        xp = new_state.xp

        P = _to_backend(projector(outcome), xp)  # (2,2)
        t = new_state.get_tensor(site)            # (chi_l, 2, chi_r)
        new_state.set_tensor(site, xp.einsum("sp,lpr->lsr", P, t))
        new_state.center = None

        # Renormalize
        norm = float(np.sqrt(abs(prob)))
        if norm > 1e-15:
            new_state.tensors[site] = new_state.tensors[site] / norm

        return new_state

    # ------------------------------------------------------------------
    # Multi-qubit measurement
    # ------------------------------------------------------------------

    def measure_qubits(
        self,
        sites: Sequence[int],
        collapse: bool = False,
    ) -> MultiQubitMeasurementResult:
        """
        Measure multiple qubits and compute the joint probability distribution.

        For m qubits, the 2^m joint probabilities are computed via the chain
        rule of conditional probabilities:

            P(b0, b1, ..., bm-1) = P(b0) · P(b1|b0) · P(b2|b0,b1) · ...

        Each factor is a single-qubit projector expectation value on the
        progressively collapsed MPS. Complexity is O(2^m · m · χ²·n).

        For large m (m > 15), use `sample_counts` instead.

        Parameters
        ----------
        sites : sequence of int
            Qubit sites to measure.
        collapse : bool
            Reserved for future use.

        Returns
        -------
        MultiQubitMeasurementResult
        """
        sites = list(sites)
        m = len(sites)
        if m == 0:
            raise ValueError("sites must be non-empty")
        if m > 20:
            raise ValueError(
                f"measure_qubits is only safe for m ≤ 20 sites (got {m}). "
                "Use sample_counts for large m."
            )

        # Enumerate all 2^m bitstrings using chain rule
        probs: Dict[Tuple[int, ...], float] = {}
        self._enumerate_joint_probs(sites, 0, self.state, (), 1.0, probs)

        # Marginals: sum over all bitstrings with the given site outcome
        marginals: Dict[int, SingleMeasurementResult] = {}
        for idx, site in enumerate(sites):
            p0 = sum(prob for bits, prob in probs.items() if bits[idx] == 0)
            p1 = sum(prob for bits, prob in probs.items() if bits[idx] == 1)
            marginals[site] = SingleMeasurementResult(site=site, prob0=p0, prob1=p1)

        return MultiQubitMeasurementResult(
            sites=sites,
            probabilities=probs,
            marginals=marginals,
        )

    def _enumerate_joint_probs(
        self,
        sites: List[int],
        depth: int,
        state: MPS,
        prefix: Tuple[int, ...],
        weight: float,
        probs: Dict[Tuple[int, ...], float],
    ):
        """
        Enumerate joint probabilities via chain rule: P(b0,...,bm) = Π P(bi | b0..bi-1).

        At each depth we compute P(outcome | current collapsed state) using a
        projector expectation value, multiply into the running weight, collapse
        the state, and recurse. At the leaf the accumulated weight IS the joint
        probability P(b0,...,bm).
        """
        if depth == len(sites):
            probs[prefix] = float(np.clip(weight, 0.0, 1.0))
            return

        site = sites[depth]
        engine = MeasurementEngine(state)

        for outcome in (0, 1):
            p_cond = engine._prob_outcome(site, outcome)
            new_weight = weight * p_cond

            if new_weight < 1e-16:
                # Zero-weight branch: fill remaining leaves with 0
                self._fill_zero(sites, depth + 1, prefix + (outcome,), probs)
                continue

            # Collapse state to outcome at this site (normalized)
            collapsed = engine._collapse_to(site, outcome, p_cond)
            # Recurse — pass un-normalized weight so leaf = joint probability
            self._enumerate_joint_probs(
                sites, depth + 1, collapsed, prefix + (outcome,), new_weight, probs
            )

    def _fill_zero(
        self,
        sites: List[int],
        depth: int,
        prefix: Tuple[int, ...],
        probs: Dict[Tuple[int, ...], float],
    ):
        """Fill all descendant bitstrings with probability 0."""
        if depth == len(sites):
            probs[prefix] = 0.0
            return
        for outcome in (0, 1):
            self._fill_zero(sites, depth + 1, prefix + (outcome,), probs)

    # ------------------------------------------------------------------
    # Corrected multi-qubit joint probability (chain rule)
    # ------------------------------------------------------------------

    def joint_probability(self, sites: Sequence[int], outcomes: Sequence[int]) -> float:
        """
        Compute a single joint probability P(b0, b1, ..., bm) for a specific
        bitstring using the chain rule:

            P(b0, b1, ..., bm) = P(b0) · P(b1|b0) · P(b2|b0,b1) · ...

        This is more efficient than enumerating all 2^m bitstrings when only
        one bitstring probability is needed.

        Parameters
        ----------
        sites : sequence of int
        outcomes : sequence of int (same length as sites, each 0 or 1)

        Returns
        -------
        float
        """
        sites = list(sites)
        outcomes = list(outcomes)
        if len(sites) != len(outcomes):
            raise ValueError("sites and outcomes must have the same length")

        state = self.state
        joint_prob = 1.0

        for site, outcome in zip(sites, outcomes):
            engine = MeasurementEngine(state)
            p = engine._prob_outcome(site, outcome)
            if p < 1e-15:
                return 0.0
            joint_prob *= p
            state = engine._collapse_to(site, outcome, p)

        return float(np.clip(joint_prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_counts(
        self,
        sites: Sequence[int],
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> SampledCounts:
        """
        Draw simulated measurement samples from the Born-rule distribution.

        Uses a sequential single-qubit collapse strategy that is efficient
        for large numbers of qubits: at each shot, measure each qubit
        sequentially, collapsing the state as we go.

        This is exact Born-rule sampling — the probabilities driving each
        sample are the true quantum probabilities, not an approximation.

        Parameters
        ----------
        sites : sequence of int
            Qubit sites to sample.
        shots : int
            Number of measurement shots.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        SampledCounts
        """
        sites = list(sites)
        rng = np.random.default_rng(seed)
        counts: Dict[str, int] = {}

        for _ in range(shots):
            state = self.state  # start fresh each shot
            bits = []

            for site in sites:
                engine = MeasurementEngine(state)
                p0 = engine._prob_outcome(site, 0)
                p0 = float(np.clip(p0, 0.0, 1.0))

                outcome = int(rng.random() >= p0)  # 0 with prob p0, 1 with prob p1
                bits.append(outcome)

                p = p0 if outcome == 0 else (1.0 - p0)
                if p < 1e-15:
                    # Degenerate: just keep state unchanged (numerically safe)
                    break
                state = engine._collapse_to(site, outcome, p)

            bs = ''.join(map(str, bits))
            counts[bs] = counts.get(bs, 0) + 1

        return SampledCounts(sites=sites, counts=counts, shots=shots)

    # ------------------------------------------------------------------
    # Full probability distribution (small systems)
    # ------------------------------------------------------------------

    def full_distribution(self, n_qubits: Optional[int] = None) -> Dict[str, float]:
        """
        Compute the full Born-rule probability distribution over all 2^n
        computational basis states.

        Requires converting MPS to statevector — only feasible for n ≤ 20.

        Parameters
        ----------
        n_qubits : int or None
            Number of qubits. Defaults to self.state.n.

        Returns
        -------
        dict mapping bitstring → probability
        """
        n = n_qubits or self.state.n
        if n > 20:
            raise ValueError(
                f"full_distribution only supports n ≤ 20 qubits (got {n}). "
                "Use sample_counts for large systems."
            )
        sv = self.state.to_statevector()
        probs = np.abs(sv) ** 2
        return {
            format(i, f'0{n}b'): float(probs[i])
            for i in range(2 ** n)
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def measure_qubit(state: MPS, site: int, collapse: bool = False) -> SingleMeasurementResult:
    """
    Measure a single qubit of an MPS state.

    Convenience wrapper around MeasurementEngine.measure_qubit.

    Parameters
    ----------
    state : MPS
    site : int
    collapse : bool

    Returns
    -------
    SingleMeasurementResult
    """
    return MeasurementEngine(state).measure_qubit(site, collapse=collapse)


def measure_qubits(
    state: MPS,
    sites: Sequence[int],
) -> MultiQubitMeasurementResult:
    """
    Measure multiple qubits of an MPS state.

    Convenience wrapper around MeasurementEngine.measure_qubits.

    Parameters
    ----------
    state : MPS
    sites : sequence of int

    Returns
    -------
    MultiQubitMeasurementResult
    """
    return MeasurementEngine(state).measure_qubits(sites)


def sample_counts(
    state: MPS,
    sites: Sequence[int],
    shots: int = 1024,
    seed: Optional[int] = None,
) -> SampledCounts:
    """
    Draw simulated measurement shots from an MPS state.

    Convenience wrapper around MeasurementEngine.sample_counts.

    Parameters
    ----------
    state : MPS
    sites : sequence of int
    shots : int
    seed : int or None

    Returns
    -------
    SampledCounts
    """
    return MeasurementEngine(state).sample_counts(sites, shots=shots, seed=seed)


def full_distribution(state: MPS) -> Dict[str, float]:
    """
    Full Born-rule probability distribution (small systems only, n ≤ 20).

    Parameters
    ----------
    state : MPS

    Returns
    -------
    dict mapping bitstring → probability
    """
    return MeasurementEngine(state).full_distribution()


__all__ = [
    # Projectors
    "projector",
    "P0",
    "P1",
    # Data classes
    "SingleMeasurementResult",
    "MultiQubitMeasurementResult",
    "SampledCounts",
    # Engine
    "MeasurementEngine",
    # Convenience
    "measure_qubit",
    "measure_qubits",
    "sample_counts",
    "full_distribution",
]
