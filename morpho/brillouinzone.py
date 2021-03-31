"""This module implements classes related to the brillouinzone."""

from abc import ABC, abstractproperty
from collections import namedtuple
from typing import List, NamedTuple, Tuple

import numpy as np
from numpy.linalg import norm


class SymmetryPoint:
    """Model of a symmetry point at the brillouinzone.

    Parameters
    ----------
    point : Tuple[float, ...]
        A point in the reciprocal domain.
    name : str
        A label to describe the point.
    """

    def __init__(self, point: Tuple[float, ...], name: str):
        """Initialize a SymmetryPoint."""
        self.point = np.array(point)
        self.name = name


class BrillouinZonePathBase(ABC):
    """BrillouinZoneBase class."""

    def __init__(self,
                 path: List[SymmetryPoint],
                 n_points: int = 50,
                 strategy: str = 'linear'):
        self.path_points = path
        self.n_points = n_points
        self.strat = strategy

    @abstractproperty
    def betas(self):
        """Return bloch wave vectors."""

    @staticmethod
    def _interpolate_beta(beta_cs, path, n_points):
        """Perform linear interpolation."""
        beta_csi = np.linspace(0, beta_cs[-1], n_points)
        beta_vi = np.vstack([
            np.interp(beta_csi, beta_cs, path[i, :]).flatten()
            for i in range(path.shape[0])
        ])
        Interp = namedtuple('InterpolatedBeta', ['values', 'cumsum'])
        return Interp(beta_vi, beta_csi)


class BrillouinZonePath1D(BrillouinZonePathBase):
    """BrillouinZonePath1D.

    Parameters
    ----------
    a1 : Tuple[float]
        Direct lattice vector a1.
    path : List[SymmetryPoint]
        List of symmetry points.
    n_points : int
        Number of vectors.
    strategy : str = 'linear'
        Strategy to interpolate.
    """

    def __init__(self,
                 a1: Tuple[float],
                 path: List[SymmetryPoint],
                 n_points: int = 50,
                 strategy: str = 'linear'):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy)
        self.a1 = np.array(a1)
        self.dim = 3

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return path[0, :] * self.b1[:, None]

    @property
    def betas(self) -> NamedTuple:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == 'linear':
            return self._interpolate_beta(beta_cs, self._path, self.n_points)
        raise NotImplementedError

    @property
    def b1(self) -> np.ndarray:
        """Return reciprocal lattice vector b1."""
        return 2 * np.pi / self.a1


class BrillouinZonePath2D(BrillouinZonePathBase):
    """BrillouinZonePath2D.

    Parameters
    ----------
    a1 : Tuple[float, float]
        Direct lattice vector a1.
    a2 : Tuple[float, float]
        Direct lattice vector a2.
    path : List[SymmetryPoint]
        List of symmetry points.
    n_points : int
        Number of vectors.
    strategy : str = 'linear'
        Strategy to interpolate.
    """

    def __init__(self,
                 a1: Tuple[float, float],
                 a2: Tuple[float, float],
                 path: List[SymmetryPoint],
                 n_points: int = 50,
                 strategy: str = 'linear'):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy)
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.dim = 3

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return (path[0, :] * self.b1[:, None] + path[1, :] * self.b2[:, None])

    @property
    def betas(self) -> NamedTuple:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == 'linear':
            return self._interpolate_beta(beta_cs, self._path, self.n_points)
        raise NotImplementedError

    @property
    def b1(self) -> np.ndarray:
        """Return reciprocal lattice vector b1."""
        Q = np.array([[0, -1], [1, 0]])
        return 2 * np.pi * Q @ self.a2 / np.dot(self.a1, Q @ self.a2)

    @property
    def b2(self) -> np.ndarray:
        """Return reciprocal lattice vector b2."""
        Q = np.array([[0, -1], [1, 0]])
        return 2 * np.pi * Q @ self.a1 / np.dot(self.a2, Q @ self.a1)


class BrillouinZonePath3D(BrillouinZonePathBase):
    """BrillouinZonePath3D.

    Parameters
    ----------
    a1 : Tuple[float, float, float]
        Direct lattice vector a1.
    a2 : Tuple[float, float, float]
        Direct lattice vector a2.
    a3 : Tuple[float, float, float]
        Direct lattice vector a3.
    path : List[SymmetryPoint]
        List of symmetry points.
    n_points : int
        Number of vectors.
    strategy : str = 'linear'
        Strategy to interpolate.
    """

    def __init__(self,
                 a1: Tuple[float, float, float],
                 a2: Tuple[float, float, float],
                 a3: Tuple[float, float, float],
                 path: List[SymmetryPoint],
                 n_points: int = 50,
                 strategy: str = 'interpolate'):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy)
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.a3 = np.array(a3)
        self.dim = 3

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return (path[0, :] * self.b1[:, None] + path[1, :] * self.b2[:, None] +
                path[2, :] * self.b3[:, None])

    @property
    def betas(self) -> NamedTuple:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == 'interpolate':
            return self._interpolate_beta(beta_cs, self._path, self.n_points)
        raise NotImplementedError

    @property
    def b1(self) -> np.ndarray:
        """Return reciprocal lattice vector b1."""
        return (2 * np.pi * np.cross(self.a2, self.a3) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def b2(self) -> np.ndarray:
        """Return reciprocal lattice vector b2."""
        return (2 * np.pi * np.cross(self.a3, self.a1) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def b3(self) -> np.ndarray:
        """Return reciprocal lattice vector b3."""
        return (2 * np.pi * np.cross(self.a1, self.a2) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))


def BrillouinZonePath(*args, **kwargs):
    """BrillouinZonePath factory."""
    dim_obj = {
        1: BrillouinZonePath1D,
        2: BrillouinZonePath2D,
        3: BrillouinZonePath3D
    }
    return dim_obj[len(args[0])](*args, **kwargs)
