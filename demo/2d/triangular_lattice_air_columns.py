"""Band structure of air columns on a triangular lattice (2D)."""

import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver, find_band_gaps, plot_bands, plot_brillouin_zone
from morpho import SymmetryPoint as SPoint


def main():
    # Textbook reference: Chapter 5, Figure 10, triangular air columns.
    a = 1.0
    radius = 0.48 * a
    eps_dielectric = 13.0
    n_grid = 129
    pwe_order = 9
    points_per_segment = 100

    a1 = (a, 0.0)
    a2 = (-a / 2, np.sqrt(3) * a / 2)
    Gamma = SPoint((0.0, 0.0), "Γ")
    M = SPoint((0.5, 0.0), "M")
    K = SPoint((1 / 3, 1 / 3), "K")
    path = BZPath(a1, a2, [Gamma, M, K, Gamma], points_per_segment=points_per_segment)

    def eps_rf(eps_r, x, y):
        eps_r[:] = eps_dielectric
        distance_squared = np.full(x.shape, np.inf)
        for i in range(-1, 2):
            for j in range(-1, 2):
                center_x = i * a1[0] + j * a2[0]
                center_y = i * a1[1] + j * a2[1]
                distance_squared = np.minimum(
                    distance_squared,
                    (x - center_x) ** 2 + (y - center_y) ** 2,
                )
        air_columns = distance_squared <= radius**2
        eps_r[air_columns] = 1.0

    geometry = Geometry(a1, a2, n_grid, n_grid, eps_rf=eps_rf)

    tm_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TM")
    te_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TE")
    tm_solution = tm_solver.run(n_bands=6, store_modes=False)
    te_solution = te_solver.run(n_bands=6, store_modes=False)

    for gap in find_band_gaps((tm_solution, te_solution)):
        print(
            f"Shared TM/TE path gap: {gap.lower:.4f}–{gap.upper:.4f}; gap-to-midgap {100 * gap.gap_to_midgap_ratio:.2f}%"
        )

    _, axis = plt.subplots(figsize=(7, 5), tight_layout=True)
    plot_bands((tm_solution, te_solution), n_bands=6, ax=axis, lw=1.4)
    plot_brillouin_zone(
        tm_solution.path, ax=axis, inset=True, inset_bounds=(0.32, 0.12, 0.30, 0.25)
    )
    axis.set_ylim(0.0, 0.8)

    plt.show()


if __name__ == "__main__":
    main()
