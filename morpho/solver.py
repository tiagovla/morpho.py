"""Implement 3D/2D solvers."""

from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional, Union

import numpy as np
from scipy.linalg import eig, eigh, solve

from .brillouinzone import BrillouinZonePath1D as BZPath1D
from .brillouinzone import BrillouinZonePath2D as BZPath2D
from .brillouinzone import BrillouinZonePath3D as BZPath3D
from .brillouinzone import BrillouinZonePathBase
from .geometry import Geometry1D, Geometry2D, Geometry3D
from .utils import convmat, harmonic_indices


def _validate_inputs(geometry, path, counts, pol=None):
    if len(counts) < 3 and pol not in ("TM", "TE"):
        raise ValueError("pol must be 'TM' or 'TE'.")
    for count in counts:
        harmonic_indices(count)
    dimension = len(counts)
    if not isinstance(geometry, (Geometry1D, Geometry2D, Geometry3D)) or not isinstance(
        path, BrillouinZonePathBase
    ):
        raise ValueError(
            "Pass a Geometry and a BrillouinZonePath of the same dimension."
        )
    if path.dim != dimension or geometry.a1.shape != (dimension,):
        raise ValueError(
            "Geometry and path must have the same dimension as the solver."
        )
    basis = np.stack([getattr(path, f"a{i + 1}") for i in range(dimension)])
    geometry_basis = np.stack(
        [getattr(geometry, f"a{i + 1}") for i in range(dimension)]
    )
    if (
        np.iscomplexobj(basis)
        or not np.isfinite(basis).all()
        or np.linalg.matrix_rank(basis) != dimension
    ):
        raise ValueError("Lattice vectors must be finite and linearly independent.")
    if not np.array_equal(geometry_basis, basis):
        raise ValueError("Geometry and path must use the same lattice vectors.")
    if len(path.path_points) < 2 or any(
        point.point.shape != (dimension,)
        or not np.isfinite(point.point).all()
        or np.iscomplexobj(point.point)
        for point in path.path_points
    ):
        raise ValueError(
            "A path needs at least two finite points of the lattice dimension."
        )
    if path.points_per_segment is None:
        if (
            not isinstance(path.n_points, Integral)
            or isinstance(path.n_points, bool)
            or path.n_points < 2
        ):
            raise ValueError("n_points must be an integer of at least two.")


def _material_samples(geometry):
    eps_r, mu_r = geometry.eps_r.copy(), geometry.mu_r.copy()
    for name, samples in (("eps_r", eps_r), ("mu_r", mu_r)):
        if not np.isfinite(samples).all() or np.any(samples == 0):
            raise ValueError(f"{name} must contain finite, nonzero material values.")
    hermitian = all(
        not np.any(samples.imag) and np.all(samples.real > 0)
        for samples in (eps_r, mu_r)
    )
    return eps_r, mu_r, hermitian


def _eigenpairs(A, B, hermitian):
    if not hermitian:
        values, vectors = eig(A, B)
        if not np.isfinite(values).all():
            raise ValueError("The material eigenproblem has nonfinite eigenvalues.")
        frequencies = np.sqrt(values.astype(complex))
        order = np.lexsort((frequencies.imag, frequencies.real))
        return frequencies[order], vectors[:, order]
    values, vectors = eigh(A, B)
    return np.sqrt(np.maximum(values, 0)), vectors


def _transverse_basis(k):
    """Span two displacement directions perpendicular to each retained k-G."""
    directions = k.T.copy()
    lengths = np.linalg.norm(directions, axis=1)
    directions[lengths == 0] = (1.0, 0.0, 0.0)
    directions /= np.where(lengths == 0, 1.0, lengths)[:, None]
    reference = np.eye(3)[np.argmin(np.abs(directions), axis=1)]
    first = np.cross(directions, reference)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(directions, first)
    count = k.shape[1]
    indices = np.arange(count)
    basis = np.zeros((3 * count, 2 * count))
    for component in range(3):
        basis[component * count + indices, indices] = first[:, component]
        basis[component * count + indices, count + indices] = second[:, component]
    return basis


@dataclass
class Solution:
    """Calculated bands and modes together with their sampling path.

    Parameters
    ----------
    path : BrillouinZonePathBase
        Path used by the solver.
    wn : numpy.ndarray, shape (n_k, n_bands)
        Unscaled eigen-wavenumbers ``omega / c``, ordered low to high.
        Lossy materials give complex frequencies, sorted by real part and then
        imaginary part. With ``exp(-i omega t)``, passive loss has Im(wn) < 0.
    modes : list of numpy.ndarray
        Eigenvectors at each sampled wave vector, with one mode per column.
    pol : str, optional
        Polarization label for 1D and 2D solutions. Vectorial 3D solutions
        have no polarization label.
    """

    path: BrillouinZonePathBase
    wn: np.ndarray
    modes: List[np.ndarray]
    pol: Optional[str] = None

    @property
    def path_distance(self) -> np.ndarray:
        """Return cumulative wave-vector distance at each sampled path point."""
        return self.path.betas.cumsum


class Solver1D:
    """Implement a 1D PWE Solver.

    Parameters
    ----------
    geometry : Geometry
        geometry
    path : BrillouinZonePath
        A BrillouinZonePath object.
    P : int
        Number of terms in the direction of the reciprocal vector b1.
    """

    def __init__(
        self, geometry: Geometry1D, path: BZPath1D, P: int = 1, pol: str = "TM"
    ):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P = P
        self.pol = pol

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self) -> Solution:
        """Calculate bands and modes and return a solution containing the path."""
        _validate_inputs(self.geo, self.path, (self.P,), self.pol)
        self.wn = []
        self.modes = []
        eps_r, mu_r, hermitian = _material_samples(self.geo)
        self.eps_rc = convmat(eps_r, self.P)
        self.mu_rc = convmat(mu_r, self.P)
        mu_rci = convmat(1 / mu_r, self.P)
        eps_rci = convmat(1 / eps_r, self.P)

        p = harmonic_indices(self.P)
        G_k = p * self.path.b1[0]
        beta = self.path.betas.values

        for col in range(beta.shape[1]):
            k = beta[0, col] - G_k
            KX = np.diag(k)

            if self.pol == "TM":
                A = KX @ mu_rci @ KX
                B = self.eps_rc
            else:
                A = KX @ eps_rci @ KX
                B = self.mu_rc

            frequencies, V = _eigenpairs(A, B, hermitian)
            self.wn.append(frequencies)
            self.modes.append(V)

        return Solution(self.path, np.stack(self.wn), self.modes.copy(), self.pol)


class Solver2D:
    """Implement a 2D PWE Solver.

    Parameters
    ----------
    geometry : Geometry
        geometry
    path : BrillouinZonePath
        A BrillouinZonePath object.
    P : int
        Number of terms in the direction of the reciprocal vector b1.
    Q : int
        Number of terms in the direction of the reciprocal vector b2.
    """

    def __init__(
        self,
        geometry: Geometry2D,
        path: BZPath2D,
        P: int = 1,
        Q: int = 1,
        pol: str = "TM",
    ):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P, self.Q = P, Q
        self.pol = pol

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self) -> Solution:
        """Calculate bands and modes and return a solution containing the path."""
        _validate_inputs(self.geo, self.path, (self.P, self.Q), self.pol)
        self.wn = []
        self.modes = []
        eps_r, mu_r, hermitian = _material_samples(self.geo)
        self.eps_rc = convmat(eps_r, self.P, self.Q)
        self.mu_rc = convmat(mu_r, self.P, self.Q)

        mu_rci = convmat(1 / mu_r, self.P, self.Q)
        if self.pol != "TM":
            # Factorize epsilon before truncating its reciprocal. This
            # inverse rule converges much faster for TE modes at sharp
            # dielectric interfaces than Fourier coefficients of 1/eps.
            eps_rci = np.linalg.inv(self.eps_rc)

        p = harmonic_indices(self.P)
        q = harmonic_indices(self.Q)

        P0, Q0 = np.meshgrid(p, q, indexing="ij")
        b1, b2 = self.path.b1, self.path.b2

        G_k = P0.ravel(order="F") * b1[:, None] + Q0.ravel(order="F") * b2[:, None]
        beta = self.path.betas.values

        for col in range(beta.shape[1]):
            k = beta[:, col, None] - G_k

            KX = np.diag(k[0, :])
            KY = np.diag(k[1, :])

            if self.pol == "TM":
                A = KX @ mu_rci @ KX + KY @ mu_rci @ KY
                B = self.eps_rc
            else:
                A = KX @ eps_rci @ KX + KY @ eps_rci @ KY
                B = self.mu_rc

            frequencies, V = _eigenpairs(A, B, hermitian)
            self.wn.append(frequencies)
            self.modes.append(V)

        return Solution(self.path, np.stack(self.wn), self.modes.copy(), self.pol)


class Solver3D:
    """Implement a 3D PWE Solver.

    Parameters
    ----------
    geometry : Geometry
        geometry
    path : BrillouinZonePath
        A BrillouinZonePath object.
    P : int
        Number of terms in the direction of the reciprocal vector b1.
    Q : int
        Number of terms in the direction of the reciprocal vector b2.
    R : int
        Number of terms in the direction of the reciprocal vector b3.
    """

    def __init__(
        self, geometry: Geometry3D, path: BZPath3D, P: int = 1, Q: int = 1, R: int = 1
    ):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P, self.Q, self.R = P, Q, R

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self) -> Solution:
        """Calculate bands and modes and return a solution containing the path."""
        _validate_inputs(self.geo, self.path, (self.P, self.Q, self.R))
        self.wn = []
        self.modes = []
        eps_r, mu_r, hermitian = _material_samples(self.geo)
        self.eps_rc = convmat(eps_r, self.P, self.Q, self.R)
        self.mu_rc = convmat(mu_r, self.P, self.Q, self.R)

        eps_rk = np.kron(np.eye(3), self.eps_rc)
        mu_ri = convmat(1 / mu_r, self.P, self.Q, self.R)
        mu_rki = np.kron(np.eye(3), mu_ri)

        p = harmonic_indices(self.P)
        q = harmonic_indices(self.Q)
        r = harmonic_indices(self.R)

        P0, Q0, R0 = np.meshgrid(p, q, r, indexing="ij")
        b1, b2, b3 = self.path.b1, self.path.b2, self.path.b3

        G_k = (
            b1[:, None] * P0.ravel(order="F")
            + b2[:, None] * Q0.ravel(order="F")
            + b3[:, None] * R0.ravel(order="F")
        )

        beta = self.path.betas.values
        for col in range(beta.shape[1]):
            k = beta[:, col, None] - G_k

            KX = np.diag(k[0, :])
            KY = np.diag(k[1, :])
            KZ = np.diag(k[2, :])
            K_V = np.vstack(
                (
                    np.hstack((0 * KX, -KZ, KY)),
                    np.hstack((KZ, 0 * KY, -KX)),
                    np.hstack((-KY, KX, 0 * KZ)),
                )
            )
            A = K_V @ mu_rki @ K_V
            B = eps_rk

            if not hermitian:
                # Constrain D = epsilon E before solving so longitudinal zeros
                # cannot be mistaken for physical complex-frequency bands.
                basis = _transverse_basis(k)
                electric_basis = solve(B, basis)
                frequencies, V = _eigenpairs(-basis.T @ A @ electric_basis, None, False)
                self.wn.append(frequencies)
                self.modes.append(electric_basis @ V)
                continue

            D, V = eigh(A, B)

            # The curl-curl operator has one longitudinal null mode per
            # retained reciprocal vector. Drop those N zero eigenvalues;
            # at Gamma the two physical zero-frequency modes remain.
            n_harmonics = KX.shape[0]
            D = D[:-n_harmonics]
            V = V[:, :-n_harmonics]

            # eigh returns the negative eigenvalues from most negative to
            # least negative, so reverse to report frequencies low to high.
            self.wn.append(np.sqrt(np.maximum(-D, 0))[::-1])
            self.modes.append(V[:, ::-1])

        return Solution(self.path, np.stack(self.wn), self.modes.copy())


def Solver(geometry, path, *args, **kwargs) -> Union[Solver1D, Solver2D, Solver3D]:
    """Solver factory."""
    dim_obj = {1: Solver1D, 2: Solver2D, 3: Solver3D}
    return dim_obj[path.dim](geometry, path, *args, **kwargs)
