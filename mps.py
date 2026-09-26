"""
Matrix Product State (MPS) core implementation.

An MPS represents an n-qubit state as a chain of rank-3 tensors:
    Γ[0][σ0, r0] · Γ[1][l1, σ1, r1] · ... · Γ[n-1][ln-1, σn-1]

where σ are physical (qubit) indices and l,r are virtual (bond) indices.

GPU support
-----------
Pass device='cuda' to MPS or MPSSimulator to run all tensor operations on
the GPU via CuPy. If CuPy is not installed or no GPU is available, the
simulator automatically falls back to CPU (NumPy) and emits a warning.

    from mps_xtrap import MPSSimulator
    sim = MPSSimulator(chi=64, device='cuda')   # GPU
    sim = MPSSimulator(chi=64, device='cpu')    # CPU (default)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_backend(device: str):
    """
    Return the array module (cupy or numpy) for the requested device.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    xp : module
        cupy if device='cuda' and CuPy is available, else numpy.
    device : str
        Resolved device string ('cuda' or 'cpu').
    """
    if device == 'cuda':
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()   # raises if no GPU present
            return cp, 'cuda'
        except Exception as e:
            logger.warning(
                f"GPU requested but not available ({e}). Falling back to CPU."
            )
            return np, 'cpu'
    return np, 'cpu'


def _to_backend(array, xp):
    """Move a numpy array to the target backend (cupy or numpy)."""
    if xp is np:
        return np.asarray(array)
    import cupy as cp
    return cp.asarray(array)


def _to_numpy(array) -> NDArray:
    """Move an array back to numpy (from cupy if needed)."""
    try:
        return array.get()   # cupy -> numpy
    except AttributeError:
        return np.asarray(array)


# ---------------------------------------------------------------------------
# MPS
# ---------------------------------------------------------------------------

class MPS:
    """
    Matrix Product State representation of an n-qubit quantum state.

    Tensors are stored in mixed-canonical form centered at a sweep position,
    which ensures numerically stable expectation value computation.

    Parameters
    ----------
    n : int
        Number of qubits.
    chi : int
        Maximum bond dimension (truncation threshold).
    device : str
        'cpu' (default) or 'cuda'. With 'cuda', all tensors are stored
        and all operations are performed on the GPU via CuPy.
        Falls back to CPU automatically if CuPy is unavailable.

    Attributes
    ----------
    tensors : list of ndarray
        MPS tensors. Boundary shapes are (2, chi_r) for i=0 and
        (chi_l, 2) for i=n-1. Interior shapes are (chi_l, 2, chi_r).
    center : int or None
        Orthogonality center site.
    truncation_errors : list of float
        Cumulative truncation errors from SVD truncations.
    device : str
        Active device ('cpu' or 'cuda').
    """

    def __init__(self, n: int, chi: int, device: str = 'cpu'):
        self.n = n
        self.chi = chi
        self.xp, self.device = _resolve_backend(device)
        self.tensors: list = []
        self.center: Optional[int] = None
        self.truncation_errors: List[float] = []
        self._initialize_product_state()

    def _initialize_product_state(self):
        """Initialize |00...0> product state on the active device."""
        xp = self.xp
        self.tensors = []
        for i in range(self.n):
            if i == 0:
                t = xp.zeros((2, 1), dtype=complex)
                t[0, 0] = 1.0
            elif i == self.n - 1:
                t = xp.zeros((1, 2), dtype=complex)
                t[0, 0] = 1.0
            else:
                t = xp.zeros((1, 2, 1), dtype=complex)
                t[0, 0, 0] = 1.0
            self.tensors.append(t)
        self.center = None
        self.truncation_errors = []

    # ------------------------------------------------------------------
    # Tensor shape helpers
    # ------------------------------------------------------------------

    def bond_dim(self, bond: int) -> int:
        """Bond dimension of bond between site bond and bond+1."""
        if bond < 0 or bond >= self.n - 1:
            raise ValueError(f"Bond {bond} out of range for {self.n} qubits.")
        t = self.tensors[bond]
        return int(t.shape[1] if bond == 0 else t.shape[2])

    def get_tensor(self, i: int):
        """Return tensor at site i with uniform shape (chi_l, 2, chi_r)."""
        t = self.tensors[i]
        xp = self.xp
        if i == 0:
            return t[xp.newaxis, :, :]   # (1, 2, chi_r)
        elif i == self.n - 1:
            return t[:, :, xp.newaxis]   # (chi_l, 2, 1)
        return t

    def set_tensor(self, i: int, t):
        """Set tensor at site i, stripping trivial boundary indices."""
        if i == 0 and t.ndim == 3:
            t = t[0, :, :]
        elif i == self.n - 1 and t.ndim == 3:
            t = t[:, :, 0]
        self.tensors[i] = t

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------

    def left_normalize_site(self, i: int):
        """Left-normalize site i via QR. Returns R to absorb into site i+1."""
        xp = self.xp
        t = self.get_tensor(i)           # (chi_l, 2, chi_r)
        chi_l, d, chi_r = t.shape
        mat = t.reshape(chi_l * d, chi_r)
        Q, R = xp.linalg.qr(mat)
        self.set_tensor(i, Q.reshape(chi_l, d, Q.shape[1]))
        return R

    def canonicalize(self, center: int):
        """
        Bring MPS to mixed-canonical form with orthogonality center at `center`.
        Sites 0..center-1 are left-normalized, sites center+1..n-1 are right-normalized.
        """
        xp = self.xp

        # Left sweep
        for i in range(center):
            R = self.left_normalize_site(i)
            t_next = self.get_tensor(i + 1)
            t_next = xp.einsum('ij,jkl->ikl', R, t_next)
            self.set_tensor(i + 1, t_next)

        # Right sweep
        for i in range(self.n - 1, center, -1):
            t = self.get_tensor(i)
            chi_l, d, chi_r = t.shape
            Q, R = xp.linalg.qr(t.reshape(chi_l, d * chi_r).T)
            new_chi = Q.shape[1]
            self.set_tensor(i, Q.T.reshape(new_chi, d, chi_r))
            t_prev = self.get_tensor(i - 1)
            self.set_tensor(i - 1, xp.einsum('ijk,kl->ijl', t_prev, R.T))

        self.center = center

    # ------------------------------------------------------------------
    # SVD-based gate application with truncation
    # ------------------------------------------------------------------

    def apply_svd_truncation(
        self,
        i: int,
        theta,
        chi: Optional[int] = None,
        svd_threshold: float = 1e-14,
    ) -> float:
        """
        Apply SVD on the combined two-site tensor theta at bond (i, i+1).

        Parameters
        ----------
        i : int
            Left site of the bond.
        theta : array, shape (chi_l, 2, 2, chi_r)
            Combined two-site tensor (on the active device).
        chi : int, optional
            Bond dimension cap. Defaults to self.chi.
        svd_threshold : float
            Singular values below this threshold are dropped.

        Returns
        -------
        float
            Truncation error (sum of squared discarded singular values).
        """
        xp = self.xp
        if chi is None:
            chi = self.chi

        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)

        U, S, Vh = xp.linalg.svd(mat, full_matrices=False)

        keep = int(xp.sum(S > svd_threshold))
        keep = min(keep, chi)
        keep = max(keep, 1)

        trunc_err = float(_to_numpy(xp.sum(S[keep:] ** 2)))
        self.truncation_errors.append(trunc_err)

        SV = xp.diag(S[:keep]) @ Vh[:keep, :]
        self.set_tensor(i,     U[:, :keep].reshape(chi_l, d1, keep))
        self.set_tensor(i + 1, SV.reshape(keep, d2, chi_r))

        return trunc_err

    def apply_block_svd(
        self,
        start: int,
        block_tensor,
        k: int,
        chi: Optional[int] = None,
        svd_threshold: float = 1e-14,
    ) -> float:
        """
        Split a combined k-site tensor into k separate MPS site tensors,
        via a left-to-right sequence of SVDs with bond-dimension truncation
        applied at every cut.

        This generalizes :meth:`apply_svd_truncation` (which handles the
        k=2 case) to blocks of any size. It is the mechanism used to apply
        fused multi-qubit gates — produced by overlapping-support gate
        fusion — that span more than two adjacent sites: the whole block
        is contracted with the fused gate into one big tensor, and this
        method peels it back apart into a valid MPS, one truncated SVD per
        internal bond, exactly as a standard two-site update does at a
        single bond.

        Parameters
        ----------
        start : int
            Leftmost site index of the block.
        block_tensor : array, shape (chi_l, 2, 2, ..., 2, chi_r)
            Combined tensor spanning sites ``start .. start + k - 1``,
            with k physical (dimension-2) legs in between the left and
            right bond legs.
        k : int
            Number of sites in the block (k >= 2).
        chi : int, optional
            Bond dimension cap for every internal cut. Defaults to self.chi.
        svd_threshold : float
            Singular values below this threshold are dropped.

        Returns
        -------
        float
            Total truncation error (sum over all k-1 internal cuts) incurred
            splitting this block.
        """
        xp = self.xp
        if chi is None:
            chi = self.chi
        if k < 2:
            raise ValueError("apply_block_svd requires k >= 2 sites.")

        d = 2
        chi_l = block_tensor.shape[0]
        chi_r = block_tensor.shape[-1]

        remaining = block_tensor
        left_bond = chi_l
        total_trunc = 0.0

        # Left-to-right sweep: peel off one site at a time, exactly like a
        # chain of two-site SVD updates.
        for site_offset in range(k - 1):
            remaining_sites = k - site_offset
            mat = remaining.reshape(left_bond * d, -1)

            U, S, Vh = xp.linalg.svd(mat, full_matrices=False)

            keep = int(xp.sum(S > svd_threshold))
            keep = min(keep, chi)
            keep = max(keep, 1)

            trunc_err = float(_to_numpy(xp.sum(S[keep:] ** 2)))
            total_trunc += trunc_err
            self.truncation_errors.append(trunc_err)

            self.set_tensor(start + site_offset, U[:, :keep].reshape(left_bond, d, keep))

            SV = xp.diag(S[:keep]) @ Vh[:keep, :]
            left_bond = keep
            remaining = SV.reshape((left_bond,) + (d,) * (remaining_sites - 1) + (chi_r,))

        # What's left after the sweep is exactly the final site's tensor.
        self.set_tensor(start + k - 1, remaining)
        return total_trunc

    # ------------------------------------------------------------------
    # Expectation values
    # ------------------------------------------------------------------

    def expectation_single(self, op, site: int) -> complex:
        """Compute <psi|op|psi> for a single-site operator."""
        xp = self.xp
        op = _to_backend(op, xp)
        self.canonicalize(site)
        t = self.get_tensor(site)
        val = xp.einsum('lsr,sa,lar->', t.conj(), op, t)
        return complex(_to_numpy(val))

    def expectation_two(self, op, site_i: int) -> complex:
        """Compute <psi|op|psi> for a two-site operator at (site_i, site_i+1)."""
        xp = self.xp
        op = _to_backend(op, xp)
        self.canonicalize(site_i)
        A = self.get_tensor(site_i)
        B = self.get_tensor(site_i + 1)
        theta = xp.einsum('lir,rjs->lijs', A, B)
        val = xp.einsum('lijs,abij,labs->', theta.conj(), op, theta)
        return complex(_to_numpy(val))

    def expectation_multi_site(self, ops, sites) -> complex:
        """
        Compute <psi| op_0(sites[0]) op_1(sites[1]) ... |psi> for any number
        of single-site operators at arbitrary, possibly non-adjacent, sites.

        This generalizes expectation_two_site to N sites using a single
        left-to-right transfer-matrix sweep: identity transfer matrices are
        used between insertion points, and the given operator is inserted
        wherever a site is specified. No truncation, no state mutation.
        Cost is O(n * chi^3), same order as a single canonicalization sweep,
        regardless of how many sites are involved.

        Parameters
        ----------
        ops : sequence of arrays
            Single-site operators (2x2), one per entry in `sites`.
        sites : sequence of int
            Site indices, same length as `ops`. Must all be distinct.
        """
        xp = self.xp
        if len(ops) != len(sites):
            raise ValueError("ops and sites must have the same length.")
        if len(sites) == 0:
            raise ValueError("Need at least one (op, site) pair.")
        if len(set(sites)) != len(sites):
            raise ValueError("All sites must be distinct.")

        # Sort by site so the sweep proceeds left to right
        order = sorted(range(len(sites)), key=lambda k: sites[k])
        sorted_sites = [sites[k] for k in order]
        sorted_ops = [_to_backend(ops[k], xp) for k in order]

        first_site = sorted_sites[0]
        self.canonicalize(first_site)

        # Left boundary: insert the first operator
        t = self.get_tensor(first_site)
        L = xp.einsum('lsr,sa,lat->rt', t.conj(), sorted_ops[0], t)

        insertion_idx = 1
        for pos in range(first_site + 1, sorted_sites[-1] + 1):
            t = self.get_tensor(pos)
            if insertion_idx < len(sorted_sites) and pos == sorted_sites[insertion_idx]:
                op = sorted_ops[insertion_idx]
                L = xp.einsum('rt,rsp,sa,taq->pq', L, t.conj(), op, t)
                insertion_idx += 1
            else:
                L = xp.einsum('rt,rsp,tsq->pq', L, t.conj(), t)

        val = xp.einsum('rr->', L)
        return complex(_to_numpy(val))

    def expectation_two_site(self, op_i, op_j, site_i: int, site_j: int) -> complex:
        """
        Compute <psi| op_i(site_i) op_j(site_j) |psi> for two arbitrary,
        possibly non-adjacent, sites.

        Thin wrapper around expectation_multi_site, kept for backward
        compatibility and as the fast path for the common two-site case.
        """
        if site_i == site_j:
            raise ValueError("site_i and site_j must be different sites.")
        return self.expectation_multi_site([op_i, op_j], [site_i, site_j])

    def expectation_pauli_z(self, site: int) -> float:
        """Compute <Z> at site."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return float(np.real(_to_numpy(self.expectation_single(Z, site))))

    def expectation_pauli_x(self, site: int) -> float:
        """Compute <X> at site."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        return float(np.real(_to_numpy(self.expectation_single(X, site))))

    def expectation_pauli_y(self, site: int) -> float:
        """Compute <Y> at site."""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return float(np.real(_to_numpy(self.expectation_single(Y, site))))

    # ------------------------------------------------------------------
    # State vector (small systems only)
    # ------------------------------------------------------------------

    def to_statevector(self) -> NDArray:
        """
        Contract MPS into full statevector. Exponential cost — use only for n <= 20.
        Always returns a NumPy array regardless of device.
        """
        if self.n > 20:
            raise ValueError("to_statevector is only safe for n <= 20 qubits.")

        xp = self.xp
        result = self.get_tensor(0)[0]   # (2, chi_r)

        for i in range(1, self.n):
            t = self.get_tensor(i)
            result = xp.einsum('...l,lsr->...sr', result, t)
            result = result.reshape(-1, t.shape[2])

        return _to_numpy(result[:, 0])

    def _norm_mps(self) -> float:
        """MPS norm via transfer matrix contraction (efficient for large n)."""
        xp = self.xp
        L = xp.ones((1, 1), dtype=complex)
        for i in range(self.n):
            t = self.get_tensor(i)
            L = xp.einsum('ab,asr,bsp->rp', L, t, t.conj())
        return float(np.real(_to_numpy(L[0, 0])) ** 0.5)

    # ------------------------------------------------------------------
    # Device mobility
    # ------------------------------------------------------------------

    def to(self, device: str) -> 'MPS':
        """
        Return a copy of this MPS moved to a different device.

        Parameters
        ----------
        device : str
            'cpu' or 'cuda'.

        Returns
        -------
        MPS
            New MPS on the target device.
        """
        new = MPS(self.n, self.chi, device=device)
        new.tensors = [_to_backend(_to_numpy(t), new.xp) for t in self.tensors]
        new.center = self.center
        new.truncation_errors = list(self.truncation_errors)
        return new

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> 'MPS':
        """Return a deep copy of this MPS on the same device."""
        new = MPS(self.n, self.chi, device=self.device)
        new.tensors = [t.copy() for t in self.tensors]
        new.center = self.center
        new.truncation_errors = list(self.truncation_errors)
        return new

    def total_truncation_error(self) -> float:
        """Sum of all squared truncation errors accumulated during simulation."""
        return sum(self.truncation_errors)

    def bond_dimensions(self) -> List[int]:
        """Return list of all bond dimensions."""
        return [self.bond_dim(i) for i in range(self.n - 1)]

    def max_bond_dim(self) -> int:
        return max(self.bond_dimensions()) if self.n > 1 else 1

    def __repr__(self) -> str:
        bonds = self.bond_dimensions()
        return (f"MPS(n={self.n}, chi={self.chi}, device='{self.device}', "
                f"bonds={bonds}, trunc_err={self.total_truncation_error():.2e})")
