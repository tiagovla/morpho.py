"""This module implements classes related to the brillouinzone."""

from abc import ABC, abstractproperty
from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm


@dataclass
class BlochVectors:
    """Represents a data structure of Bloch wave vectors."""

    values: np.ndarray
    cumsum: np.ndarray


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

    def __init__(
        self,
        path: List[SymmetryPoint],
        n_points: int = 50,
        strategy: str = "linear",
        points_per_segment: Optional[int] = None,
    ):
        self.path_points = path
        self.n_points = n_points
        self.strat = strategy
        if points_per_segment is not None:
            if (
                not isinstance(points_per_segment, Integral)
                or isinstance(points_per_segment, bool)
                or points_per_segment < 1
            ):
                raise ValueError("points_per_segment must be a positive integer.")
            if len(path) < 2:
                raise ValueError(
                    "At least two symmetry points are needed for segment sampling."
                )
            points_per_segment = int(points_per_segment)
        self.points_per_segment = points_per_segment
        self._path: np.ndarray

    @abstractproperty
    def betas(self) -> BlochVectors:
        """Return bloch wave vectors."""

    @staticmethod
    def _interpolate_beta(beta_cs, path, n_points, points_per_segment=None):
        """Interpolate a path globally or by segment."""
        if points_per_segment is not None:
            beta_values = []
            beta_distances = []
            for index in range(path.shape[1] - 1):
                fractions = np.linspace(0.0, 1.0, points_per_segment + 1)
                start = path[:, index, None]
                delta = (path[:, index + 1] - path[:, index])[:, None]
                segment_values = start + delta * fractions[None, :]

                start_distance = beta_cs[index]
                segment_length = beta_cs[index + 1] - start_distance
                segment_distances = start_distance + segment_length * fractions

                if index:
                    segment_values = segment_values[:, 1:]
                    segment_distances = segment_distances[1:]
                beta_values.append(segment_values)
                beta_distances.append(segment_distances)

            return BlochVectors(
                np.concatenate(beta_values, axis=1),
                np.concatenate(beta_distances),
            )

        beta_csi = np.linspace(0, beta_cs[-1], n_points)
        beta_vi = np.vstack(
            [
                np.interp(beta_csi, beta_cs, path[i, :]).flatten()
                for i in range(path.shape[0])
            ]
        )
        return BlochVectors(beta_vi, beta_csi)

    @property
    def point_locations(self):
        """Return cumulative sum of vectors' length."""
        return np.concatenate(
            (np.zeros(1), np.cumsum(norm(np.diff(self._path, axis=1), axis=0)))
        )

    @property
    def point_names(self):
        """Return list of symmetry point names."""
        return [p.name for p in self.path_points]


class BrillouinZonePath1D(BrillouinZonePathBase):
    """BrillouinZonePath1D.

    Parameters
    ----------
    a1 : Tuple[float]
        Direct lattice vector a1.
    path : List[SymmetryPoint]
        List of symmetry points.
    n_points : int
        Total number of vectors sampled along the full path. Used when
        ``points_per_segment`` is not set.
    strategy : str = 'linear'
        Strategy to interpolate.
    points_per_segment : int, optional
        Number of intervals on each segment. Every symmetry point is sampled;
        the total is one plus this value times the number of segments. This
        takes precedence over ``n_points``.
    """

    def __init__(
        self,
        a1: Tuple[float],
        path: List[SymmetryPoint],
        n_points: int = 50,
        strategy: str = "linear",
        points_per_segment: Optional[int] = None,
    ):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy, points_per_segment)
        self.a1 = np.array(a1)
        self.dim = 1

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return path[0, :] * self.b1[:, None]

    @property
    def betas(self) -> BlochVectors:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == "linear":
            return self._interpolate_beta(
                beta_cs, self._path, self.n_points, self.points_per_segment
            )
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
        Total number of vectors sampled along the full path. Used when
        ``points_per_segment`` is not set.
    strategy : str = 'linear'
        Strategy to interpolate.
    points_per_segment : int, optional
        Number of intervals on each segment. Every symmetry point is sampled;
        the total is one plus this value times the number of segments. This
        takes precedence over ``n_points``.
    """

    def __init__(
        self,
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        path: List[SymmetryPoint],
        n_points: int = 50,
        strategy: str = "linear",
        points_per_segment: Optional[int] = None,
    ):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy, points_per_segment)
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.dim = 2

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return path[0, :] * self.b1[:, None] + path[1, :] * self.b2[:, None]

    @property
    def betas(self) -> BlochVectors:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == "linear":
            return self._interpolate_beta(
                beta_cs, self._path, self.n_points, self.points_per_segment
            )
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
        Total number of vectors sampled along the full path. Used when
        ``points_per_segment`` is not set.
    strategy : str = 'linear'
        Strategy to interpolate.
    points_per_segment : int, optional
        Number of intervals on each segment. Every symmetry point is sampled;
        the total is one plus this value times the number of segments. This
        takes precedence over ``n_points``.
    """

    def __init__(
        self,
        a1: Tuple[float, float, float],
        a2: Tuple[float, float, float],
        a3: Tuple[float, float, float],
        path: List[SymmetryPoint],
        n_points: int = 50,
        strategy: str = "interpolate",
        points_per_segment: Optional[int] = None,
    ):
        """Initialize a BrillouinZonePath."""
        super().__init__(path, n_points, strategy, points_per_segment)
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.a3 = np.array(a3)
        self.dim = 3

    @property
    def _path(self) -> np.ndarray:
        """Return the path matrix."""
        path = np.stack([p.point for p in self.path_points], axis=1)
        return (
            path[0, :] * self.b1[:, None]
            + path[1, :] * self.b2[:, None]
            + path[2, :] * self.b3[:, None]
        )

    @property
    def betas(self) -> BlochVectors:
        """Return beta vector values and cumsum."""
        beta_cs = np.cumsum(norm(np.diff(self._path, axis=1), axis=0))
        beta_cs = np.pad(beta_cs, (1, 0), "constant")
        if self.strat == "interpolate":
            return self._interpolate_beta(
                beta_cs, self._path, self.n_points, self.points_per_segment
            )
        raise NotImplementedError

    @property
    def b1(self) -> np.ndarray:
        """Return reciprocal lattice vector b1."""
        return (
            2
            * np.pi
            * np.cross(self.a2, self.a3)
            / np.dot(self.a1, np.cross(self.a2, self.a3))
        )

    @property
    def b2(self) -> np.ndarray:
        """Return reciprocal lattice vector b2."""
        return (
            2
            * np.pi
            * np.cross(self.a3, self.a1)
            / np.dot(self.a1, np.cross(self.a2, self.a3))
        )

    @property
    def b3(self) -> np.ndarray:
        """Return reciprocal lattice vector b3."""
        return (
            2
            * np.pi
            * np.cross(self.a1, self.a2)
            / np.dot(self.a1, np.cross(self.a2, self.a3))
        )


def BrillouinZonePath(
    a1, *args, **kwargs
) -> Union[BrillouinZonePath1D, BrillouinZonePath2D, BrillouinZonePath3D]:
    """BrillouinZonePath factory."""
    dim_obj = {1: BrillouinZonePath1D, 2: BrillouinZonePath2D, 3: BrillouinZonePath3D}
    return dim_obj[len(a1)](a1, *args, **kwargs)
