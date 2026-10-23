"""Shared lattice and geometry helpers for the fcc examples."""

from itertools import product

import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import SymmetryPoint as SPoint


def primitive_vectors(a):
    """Return fcc primitive vectors for a cubic supercell of side ``a``."""
    return (
        (a / 2, a / 2, 0.0),
        (0.0, a / 2, a / 2),
        (a / 2, 0.0, a / 2),
    )


def symmetry_path(a1, a2, a3, points_per_segment=20):
    """Follow X-U-L-Γ-X-W-K using reciprocal primitive coordinates."""
    points = [
        SPoint((1 / 2, 1 / 2, 0.0), "X"),
        SPoint((5 / 8, 5 / 8, 1 / 4), "U"),
        SPoint((1 / 2, 1 / 2, 1 / 2), "L"),
        SPoint((0.0, 0.0, 0.0), "Γ"),
        SPoint((1 / 2, 1 / 2, 0.0), "X"),
        SPoint((3 / 4, 1 / 2, 1 / 4), "W"),
        SPoint((3 / 4, 3 / 8, 3 / 8), "K"),
    ]
    return BZPath(a1, a2, a3, points, points_per_segment=points_per_segment)


def periodic_sphere_mask(x, y, z, lattice_vectors, radius, basis_offsets):
    """Mark points inside spheres at every site of a periodic fcc basis."""
    a1, a2, a3 = (np.asarray(vector) for vector in lattice_vectors)
    distance_squared = np.full(x.shape, np.inf)
    for i, j, k in product((-1, 0, 1), repeat=3):
        translation = i * a1 + j * a2 + k * a3
        for offset in basis_offsets:
            center = translation + np.asarray(offset)
            distance_squared = np.minimum(
                distance_squared,
                (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2,
            )
    return distance_squared <= radius**2
