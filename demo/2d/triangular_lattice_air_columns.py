"""Band structure of air columns on a triangular lattice (2D)."""

import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
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
    path = BZPath(
        a1, a2, [Gamma, M, K, Gamma], points_per_segment=points_per_segment
    )
    geometry = Geometry(a1, a2, n_grid, n_grid)

    @geometry.overwrite
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

    tm_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TM")
    te_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TE")
    tm_solver.run()
    te_solver.run()

    scale = a / (2 * np.pi)
    tm = np.vstack(tm_solver.wn) * scale
    te = np.vstack(te_solver.wn) * scale
    path_distance = path.betas.cumsum

    gap_lower = max(np.max(tm[:, 1]), np.max(te[:, 0]))
    gap_upper = min(np.min(tm[:, 2]), np.min(te[:, 1]))
    gap_ratio = 200.0 * (gap_upper - gap_lower) / (gap_upper + gap_lower)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.axhspan(gap_lower, gap_upper, color="#ffe600", alpha=0.8, zorder=0)
    band_count = min(6, tm.shape[1], te.shape[1])
    for band in range(band_count):
        axis.plot(path_distance, tm[:, band], color="tab:blue", lw=1.4)
        axis.plot(path_distance, te[:, band], color="tab:red", lw=1.4)
    axis.text(
        path_distance[-1] * 0.15,
        0.12,
        "TM modes",
        color="tab:blue",
        transform=axis.get_xaxis_transform(),
        va="center",
    )
    axis.text(
        path_distance[-1] * 0.05,
        0.42,
        "TE modes",
        color="tab:red",
        transform=axis.get_xaxis_transform(),
        va="center",
    )
    axis.text(
        path_distance[-1] * 0.55,
        (gap_lower + gap_upper) / 2,
        "Photonic Band Gap",
        ha="center",
        va="center",
    )
    axis.set_xticks(path.point_locations)
    axis.set_xticklabels(path.point_names)
    axis.set_xlim(0.0, path_distance[-1])
    axis.set_ylim(0.0, 0.8)
    axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
    axis.set_xlabel(r"Bloch wave vector $k$")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    print(f"TM band 2 maximum along Γ-M-K-Γ: {np.max(tm[:, 1]):.6f}")
    print(f"TE band 1 maximum along Γ-M-K-Γ: {np.max(te[:, 0]):.6f}")
    print(f"TM band 3 minimum along Γ-M-K-Γ: {np.min(tm[:, 2]):.6f}")
    print(f"TE band 2 minimum along Γ-M-K-Γ: {np.min(te[:, 1]):.6f}")
    print(
        f"Complete gap-to-midgap ratio: {gap_ratio:.2f}% "
        "(book reference: 18.6%)"
    )
    plt.show()


if __name__ == "__main__":
    main()
