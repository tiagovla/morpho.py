"""This module implements classes related to the brillouinzone."""

from typing import List, Tuple

import numpy as np
from numpy.linalg import norm


class SymmetryPoint:
    """Model of a symmetry point at the brillouinzone."""

    def __init__(self, point: Tuple[float, float, float], name: str):
        """Initialize a SymmetryPoint."""
        self.point = np.array(point).reshape(-1, 1)
        self.name = name


class BrillouinZonePath:
    """Model of a path at the brillouinzone."""

    def __init__(
        self,
        path: List[SymmetryPoint],
        t1: Tuple[float, float, float],
        t2: Tuple[float, float, float],
        t3: Tuple[float, float, float],
        n_points: int = 5,
    ):
        """Initialize a BrillouinZonePath."""
        self.path_points = path
        self.n_points = n_points
        self.t1 = np.array(t1).reshape(-1, 1)
        self.t2 = np.array(t2).reshape(-1, 1)
        self.t3 = np.array(t3).reshape(-1, 1)

        self.T1 = (2 * np.pi * np.cross(self.t2, self.t3, axis=0) /
                   np.dot(t1, np.cross(self.t2, self.t3, axis=0)))
        self.T2 = (2 * np.pi * np.cross(self.t3, self.t1, axis=0) /
                   np.dot(t1, np.cross(self.t2, self.t3, axis=0)))
        self.T3 = (2 * np.pi * np.cross(self.t1, self.t2, axis=0) /
                   np.dot(t1, np.cross(self.t2, self.t3, axis=0)))

        beta_path = np.stack([p.point for p in self.path_points], axis=1)
        beta_path = (beta_path[0, :].flatten() * self.T1 +
                     beta_path[1, :].flatten() * self.T2 +
                     beta_path[2, :].flatten() * self.T3)

        beta_len = np.cumsum(norm(np.diff(beta_path, axis=1), axis=0))
        beta_len = np.hstack([0, beta_len])
        beta_len_interp = np.linspace(0, beta_len[-1], self.n_points)
        beta_ix = np.interp(beta_len_interp, beta_len,
                            beta_path[0, :].flatten())
        beta_iy = np.interp(beta_len_interp, beta_len,
                            beta_path[1, :].flatten())
        beta_iz = np.interp(beta_len_interp, beta_len,
                            beta_path[2, :].flatten())
        self.beta_vec = np.vstack((beta_ix, beta_iy, beta_iz))
        self.beta_vec_len = beta_len_interp
        self.symmetry_names = [p.name for p in self.path_points]
        self.symmetry_locations = beta_len
