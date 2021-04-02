"""Implement 3D/2D solvers."""

from typing import List, Union

import numpy as np
from scipy.linalg import eigh

from .brillouinzone import BrillouinZonePath1D as BZPath1D
from .brillouinzone import BrillouinZonePath2D as BZPath2D
from .brillouinzone import BrillouinZonePath3D as BZPath3D
from .geometry import Geometry1D, Geometry2D, Geometry3D
from .utils import convmat


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

    def __init__(self,
                 geometry: Geometry1D,
                 path: BZPath1D,
                 P: int = 1,
                 pol: str = "TM"):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P = P
        self.pol = pol

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self):
        """Calculate the eigen-wavenumbers and modes."""
        raise NotImplementedError


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

    def __init__(self,
                 geometry: Geometry2D,
                 path: BZPath2D,
                 P: int = 1,
                 Q: int = 1,
                 pol: str = "TM"):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P, self.Q = P, Q
        self.pol = pol

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self):
        """Calculate the eigen-wavenumbers and modes."""
        self.eps_rc = convmat(self.geo.eps_r, self.P, self.Q)
        self.mu_rc = convmat(self.geo.mu_r, self.P, self.Q)

        mu_rci = np.linalg.inv(self.mu_rc)
        eps_rci = np.linalg.inv(self.eps_rc)

        p = np.arange(-(self.P // 2), self.P // 2 + 1)
        q = np.arange(-(self.Q // 2), self.Q // 2 + 1)

        P0, Q0 = np.meshgrid(p, q)
        b1, b2 = self.path.b1, self.path.b2

        G_k = P0.flatten() * b1[:, None] + Q0.flatten() * b2[:, None]
        beta = self.path.betas.values

        for col in range(beta.shape[1]):
            k = beta[:, col, None] - G_k

            KX = np.diag(k[0, :])
            KY = np.diag(k[1, :])

            if self.pol == "TM":
                A = KX@mu_rci@KX + KY@mu_rci@KY
                B = self.eps_rc
            else:
                A = KX@eps_rci@KX + KY@eps_rci@KY
                B = self.mu_rc

            D, V = eigh(A, B)

            self.wn.append(np.sqrt(np.maximum(D, 0)))
            self.modes.append(V)


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

    def __init__(self,
                 geometry: Geometry3D,
                 path: BZPath3D,
                 P: int = 1,
                 Q: int = 1,
                 R: int = 1):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P, self.Q, self.R = P, Q, R

        self.eps_rc: np.ndarray
        self.mu_rc: np.ndarray
        self.wn: List[np.ndarray] = []
        self.modes: List[np.ndarray] = []

    def run(self):
        """Calculate the eigen-wavenumbers and modes."""
        self.eps_rc = convmat(self.geo.eps_r, self.P, self.Q, self.R)
        self.mu_rc = convmat(self.geo.mu_r, self.P, self.Q, self.R)

        eps_rk = np.kron(np.eye(3), self.eps_rc)
        mu_rki = np.kron(np.eye(3), np.linalg.inv(self.mu_rc))

        p = np.arange(-(self.P // 2), self.P // 2 + 1)
        q = np.arange(-(self.Q // 2), self.Q // 2 + 1)
        r = np.arange(-(self.R // 2), self.R // 2 + 1)

        P0, Q0, R0 = np.meshgrid(p, q, r)
        b1, b2, b3 = self.path.b1, self.path.b2, self.path.b3

        G_k = (b1[:, None] * P0.flatten() + b2[:, None] * Q0.flatten() +
               b3[:, None] * R0.flatten())

        beta = self.path.betas.values
        for col in range(beta.shape[1]):
            k = beta[:, col, None] - G_k

            KX = np.diag(k[0, :])
            KY = np.diag(k[1, :])
            KZ = np.diag(k[2, :])
            K_V = np.vstack((
                np.hstack((0 * KX, -KZ, KY)),
                np.hstack((KZ, 0 * KY, -KX)),
                np.hstack((-KY, KX, 0 * KZ)),
            ))
            A = K_V @ mu_rki @ K_V
            B = eps_rk

            D, V = eigh(A, B)

            self.wn.append(np.sqrt(-np.minimum(D, 0)))
            self.modes.append(V)


def Solver(geometry, path, *args,
           **kwargs) -> Union[Solver1D, Solver2D, Solver3D]:
    """Solver factory."""
    dim_obj = {1: Solver1D, 2: Solver2D, 3: Solver3D}
    return dim_obj[path.dim](geometry, path, *args, **kwargs)
