"""Band structure and gap of air spheres on a diamond lattice (3D)."""

import matplotlib.pyplot as plt

from _fcc import periodic_sphere_mask, primitive_vectors, symmetry_path
from morpho import Geometry, Solver, plot_bands, plot_brillouin_zone


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
    solution = solver.run()

    _, axis = plt.subplots(figsize=(6, 5), tight_layout=True)
    plot_bands(solution, n_bands=32, ax=axis, lattice_constant=a, color="navy", lw=0.8)
    zone_axis = plot_brillouin_zone(
        solution.path, ax=axis, inset=True, inset_bounds=(0.0, 0.0, 0.36, 0.30)
    )
    zone_axis.view_init(elev=25, azim=-105)
    axis.set_ylim(0.0, 1.0)

    print("Structure: diamond lattice of air spheres in epsilon=13")
    print(f"Sphere radius: {radius / a:.3f}a")
    print(f"Plane-wave cutoff: {pwe_order} x {pwe_order} x {pwe_order}")
    plt.show()


if __name__ == "__main__":
    main()
