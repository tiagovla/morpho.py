"""Band structure of close-packed dielectric spheres on an fcc lattice."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver, plot_bands, plot_brillouin_zone
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


def main():
    # Textbook reference: Chapter 6, Figure 2, close-packed dielectric spheres.
    # The cubic supercell lattice constant is normalized to a=1.
    a = 1.0
    a1, a2, a3 = primitive_vectors(a)
    path = symmetry_path(a1, a2, a3, points_per_segment=20)

    # Close-packed spheres have diameter equal to the fcc nearest-neighbor
    # distance a/sqrt(2).  They have epsilon=13 and are embedded in air.
    n_grid = 49
    pwe_order = 7
    radius = a / (2 * np.sqrt(2))
    eps_sphere = 13.0
    geometry = Geometry(a1, a2, a3, n_grid, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y, z):
        sphere = periodic_sphere_mask(x, y, z, (a1, a2, a3), radius, ((0.0, 0.0, 0.0),))
        eps_r[sphere] = eps_sphere

    solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, R=pwe_order)
    solution = solver.run()

    _, axis = plt.subplots(figsize=(6, 5), tight_layout=True)
    plot_bands(solution, n_bands=64, ax=axis, lattice_constant=a, color="navy", lw=0.8)
    zone_axis = plot_brillouin_zone(
        solution.path, ax=axis, inset=True, inset_bounds=(0.0, 0.0, 0.36, 0.30)
    )
    zone_axis.view_init(elev=25, azim=-105)
    axis.set_ylim(0.0, 1.0)

    print("Structure: fcc close-packed spheres, epsilon=13 in air")
    print(f"Plane-wave cutoff: {pwe_order} x {pwe_order} x {pwe_order}")
    print(f"Sampled {len(solution.wn)} points on X-U-L-Γ-X-W-K")
    print("The book reports no complete photonic band gap for this structure.")
    plt.show()


if __name__ == "__main__":
    main()
