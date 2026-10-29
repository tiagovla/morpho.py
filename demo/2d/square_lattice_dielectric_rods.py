"""Band structure of dielectric rods on a square lattice (2D)."""

import matplotlib.pyplot as plt

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver, plot_bands, plot_brillouin_zone
from morpho import SymmetryPoint as SPoint


def main():
    # Textbook reference: Chapter 5, Figure 2, alumina rods in air.
    a = 1.0
    radius = 0.2 * a
    eps_rod = 8.9
    n_grid = 129
    pwe_order = 9
    points_per_segment = 100

    a1, a2 = (a, 0.0), (0.0, a)
    Gamma = SPoint((0.0, 0.0), "Γ")
    X = SPoint((0.5, 0.0), "X")
    M = SPoint((0.5, 0.5), "M")
    symmetry_points = [Gamma, X, M, Gamma]

    path = BZPath(a1, a2, symmetry_points, points_per_segment=points_per_segment)
    geometry = Geometry(a1, a2, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y):
        mask = x**2 + y**2 <= radius**2
        eps_r[mask] = eps_rod

    tm_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TM")
    te_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TE")
    tm_solution = tm_solver.run()
    te_solution = te_solver.run()

    _, axis = plt.subplots(figsize=(7, 5), tight_layout=True)
    plot_bands((tm_solution, te_solution), n_bands=4, ax=axis, lw=1.4)
    plot_brillouin_zone(
        tm_solution.path, ax=axis, inset=True, inset_bounds=(0.32, 0.12, 0.30, 0.25)
    )
    axis.set_ylim(0.0, 0.8)
    plt.show()


if __name__ == "__main__":
    main()
