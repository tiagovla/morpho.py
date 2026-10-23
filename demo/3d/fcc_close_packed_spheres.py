"""Band structure of close-packed dielectric spheres on an fcc lattice."""

import matplotlib.pyplot as plt
import numpy as np

from _fcc import periodic_sphere_mask, primitive_vectors, symmetry_path
from morpho import Geometry, Solver


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
    solver.run()

    frequencies = np.vstack(solver.wn) * a / (2 * np.pi)
    path_distance = path.betas.cumsum
    point_locations = path.point_locations

    figure, axis = plt.subplots(figsize=(6, 5))
    for band in range(min(64, frequencies.shape[1])):
        axis.plot(path_distance, frequencies[:, band], color="navy", lw=0.8)
    axis.set_xticks(point_locations)
    axis.set_xticklabels(path.point_names)
    axis.set_xlim(0.0, path_distance[-1])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
    axis.set_xlabel(r"Bloch wave vector $k$")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()

    print("Structure: fcc close-packed spheres, epsilon=13 in air")
    print(f"Plane-wave cutoff: {pwe_order} x {pwe_order} x {pwe_order}")
    print(f"Sampled {len(path_distance)} points on X-U-L-Γ-X-W-K")
    print("The book reports no complete photonic band gap for this structure.")
    plt.show()


if __name__ == "__main__":
    main()
