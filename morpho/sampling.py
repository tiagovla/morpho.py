"""Arbitrary Bloch-vector batches and uniform reciprocal-cell meshes."""

from numbers import Integral

import numpy as np

from .brillouinzone import BlochVectors


class BlochPoints:
    """Unordered Bloch-vector samples in a one-, two-, or three-dimensional cell.

    Parameters
    ----------
    *lattice_vectors : tuple of float
        Direct lattice vectors, one per spatial dimension.
    points : array-like, shape (n_samples, dimension)
        Sampling coordinates. A flat sequence is also accepted in 1D.
    fractional : bool, default=True
        Express points in reciprocal-lattice coordinates. When false, use
        Cartesian wave vectors in inverse length units, including ``2 pi``.
    """

    def __init__(self, *lattice_vectors, points, fractional=True):
        dimension = len(lattice_vectors)
        vectors = np.asarray(lattice_vectors)
        if not np.iscomplexobj(vectors):
            try:
                vectors = np.array(vectors, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(
                    "Lattice vectors must be real numeric values."
                ) from None
        if (
            dimension not in (1, 2, 3)
            or vectors.shape != (dimension, dimension)
            or np.iscomplexobj(vectors)
            or not np.isfinite(vectors).all()
            or np.linalg.matrix_rank(vectors) != dimension
        ):
            raise ValueError(
                "Supply finite, real, linearly independent lattice vectors."
            )
        if not isinstance(fractional, bool):
            raise ValueError("fractional must be a boolean.")
        coordinates = np.asarray(points)
        if np.iscomplexobj(coordinates):
            raise ValueError("Bloch points must be finite and real.")
        coordinates = np.array(coordinates, dtype=float, copy=True)
        if dimension == 1 and coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != dimension
            or coordinates.shape[0] == 0
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError(
                "Bloch points must have shape (n_samples, dimension), with at least one finite point."
            )
        reciprocal = 2 * np.pi * np.linalg.inv(vectors)
        self.dim = dimension
        self.grid_shape = None
        for index in range(dimension):
            direct = np.array(vectors[index], dtype=float, copy=True)
            dual = reciprocal[:, index].copy()
            direct.setflags(write=False)
            dual.setflags(write=False)
            setattr(self, f"a{index + 1}", direct)
            setattr(self, f"b{index + 1}", dual)
        values = reciprocal @ coordinates.T if fractional else coordinates.T.copy()
        values.setflags(write=False)
        self._betas = BlochVectors(values)

    @property
    def betas(self) -> BlochVectors:
        """Return Cartesian samples; unordered points have no path distance."""
        return self._betas


def brillouin_zone_mesh(*lattice_vectors, shape) -> BlochPoints:
    """Sample one primitive reciprocal cell on a uniform, Gamma-centered mesh.

    Parameters
    ----------
    *lattice_vectors : tuple of float
        Direct lattice vectors.
    shape : int or tuple of int
        Samples per reciprocal direction. One integer applies to every axis.

    Returns
    -------
    BlochPoints
        Uniformly weighted points covering one reciprocal primitive cell,
        without duplicated periodic endpoints. ``grid_shape`` records the
        grid dimensions. Samples flatten in C order and always include Gamma.
        For oblique lattices this cell is not the Wigner-Seitz polyhedron;
        its points cover the Brillouin zone modulo reciprocal translations.
    """
    dimension = len(lattice_vectors)
    if dimension not in (1, 2, 3):
        raise ValueError("Supply one, two, or three direct lattice vectors.")
    if isinstance(shape, Integral) and not isinstance(shape, bool):
        shape = (shape,) * dimension
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != dimension
        or any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 1
            for value in shape
        )
    ):
        raise ValueError(
            "shape must contain a positive integer for every lattice direction."
        )
    shape = tuple(int(value) for value in shape)
    axes = [(np.arange(count) - count // 2) / count for count in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack(grids, axis=-1).reshape(-1, dimension)
    sampling = BlochPoints(*lattice_vectors, points=coordinates)
    sampling.grid_shape = shape
    return sampling
