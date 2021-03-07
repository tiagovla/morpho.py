"""This module implements classes related to the brillouinzone."""

from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm


class SymmetryPoint:
    """Model of a symmetry point at the brillouinzone."""

    def __init__(self, point: Tuple[float, ...], name: str):
        """Initialize a SymmetryPoint."""
        self.point = np.array(point)
        self.name = name


class BrillouinZonePath:
    """Model of a path at the brillouinzone."""

    def __init__(
        self,
        path: List[SymmetryPoint],
        t1: Tuple[float, ...],
        t2: Optional[Tuple[float, ...]] = None,
        t3: Optional[Tuple[float, ...]] = None,
        n_points: int = 5,
    ):
        """Initialize a BrillouinZonePath."""
        self.path_points = path
        self.n_points = n_points

        self.dim = len(t1)

        self._t1 = np.pad(np.array(t1), (0, 3 - len(t1)), "constant")
        self._t2 = (np.pad(np.array(t2), (0, 3 - len(t2)), "constant")
                    if t2 else np.array([0, 1, 0]))
        self._t3 = (np.pad(np.array(t3), (0, 3 - len(t1)), "constant")
                    if t3 else np.array([0, 0, 1]))

        self._T1 = (2 * np.pi * np.cross(self._t2, self._t3) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))
        self._T2 = (2 * np.pi * np.cross(self._t3, self._t1) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))
        self._T3 = (2 * np.pi * np.cross(self._t1, self._t2) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))

        self.beta_vec = None
        self.beta_vec_len = None
        self.symmetry_names = None
        self.symmetry_locations = None
        self._calculate_beta()

    @property
    def t1(self):
        """Return vector t1."""
        return self._t1[:self.dim]

    @property
    def t2(self):
        """Return vector t2."""
        return self._t2[:self.dim]

    @property
    def t3(self):
        """Return vector t3."""
        return self._t3[:self.dim]

    @property
    def T1(self):
        """Return vector T1."""
        return self._T1[:self.dim]

    @property
    def T2(self):
        """Return vector T2."""
        return self._T2[:self.dim]

    @property
    def T3(self):
        """Return vector T3."""
        return self._T3[:self.dim]

    def _calculate_beta(self):
        beta_path = np.stack(
            [
                np.pad(p.point, (0, 3 - len(p.point)), "constant")
                for p in self.path_points
            ],
            axis=1,
        )
        beta_path = (beta_path[0, :] * self._T1[:, None] +
                     beta_path[1, :] * self._T2[:, None] +
                     beta_path[2, :] * self._T3[:, None])

        beta_len = np.cumsum(
            norm(np.diff(beta_path, axis=1)[:self.dim], axis=0))
        beta_len = np.pad(beta_len, (1, 0), "constant")
        beta_len_interp = np.linspace(0, beta_len[-1], self.n_points)
        beta_ix = np.interp(beta_len_interp, beta_len,
                            beta_path[0, :].flatten())
        beta_iy = np.interp(beta_len_interp, beta_len,
                            beta_path[1, :].flatten())
        beta_iz = np.interp(beta_len_interp, beta_len,
                            beta_path[2, :].flatten())

        self.beta_vec = np.vstack((beta_ix, beta_iy, beta_iz))[:self.dim, :]
        self.beta_vec_len = beta_len_interp
        self.symmetry_names = [p.name for p in self.path_points]
        self.symmetry_locations = beta_len
