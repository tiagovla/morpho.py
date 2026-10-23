"""Band structure and gap of air spheres on a diamond lattice (3D)."""
import matplotlib.pyplot as plt
import numpy as np

from _fcc import periodic_sphere_mask, primitive_vectors, symmetry_path
from morpho import Geometry, Solver


def main():
    # Textbook reference: Chapter 6, Figure 3, air spheres in dielectric.
    # The conventional cubic supercell lattice constant is normalized to a=1.
    a = 1.0
    a1, a2, a3 = primitive_vectors(a)
    path = symmetry_path(a1, a2, a3, points_per_segment=20)

    # Two interpenetrating fcc sublattices form the diamond lattice.  The
    # radius 0.325a makes neighboring air spheres overlap.
    n_grid = 49
    pwe_order = 7
    radius = 0.325 * a
    eps_dielectric = 13.0
    basis = ((0.0, 0.0, 0.0), (a / 4, a / 4, a / 4))
    geometry = Geometry(a1, a2, a3, n_grid, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y, z):
        eps_r[:] = eps_dielectric
        air = periodic_sphere_mask(x, y, z, (a1, a2, a3), radius, basis)
        eps_r[air] = 1.0

    solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, R=pwe_order)
    solver.run()

    frequencies = np.vstack(solver.wn) * a / (2 * np.pi)
    path_distance = path.betas.cumsum
    point_locations = path.point_locations
    lower_edge = np.max(frequencies[:, 1])
    upper_edge = np.min(frequencies[:, 2])
    gap_ratio = 200 * (upper_edge - lower_edge) / (upper_edge + lower_edge)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.axhspan(lower_edge, upper_edge, color="#ffe600", alpha=0.9, zorder=0)
    for band in range(min(32, frequencies.shape[1])):
        axis.plot(path_distance, frequencies[:, band], color="navy", lw=0.8)
    axis.text(
        path_distance[-1] / 2,
        (lower_edge + upper_edge) / 2,
        "Photonic Band Gap",
        ha="center",
        va="center",
    )
    axis.set_xticks(point_locations)
    axis.set_xticklabels(path.point_names)
    axis.set_xlim(0.0, path_distance[-1])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
    axis.set_xlabel(r"Bloch wave vector $k$")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()

    print("Structure: diamond lattice of air spheres in epsilon=13")
    print(f"Sphere radius: {radius / a:.3f}a")
    print(f"Plane-wave cutoff: {pwe_order} x {pwe_order} x {pwe_order}")
    print(f"Second-band maximum on path: {lower_edge:.5f}")
    print(f"Third-band minimum on path: {upper_edge:.5f}")
    print(
        f"Gap-to-midgap ratio on path: {gap_ratio:.2f}% "
        "(book reference: 29.6%)"
    )
    plt.show()


if __name__ == "__main__":
    main()
