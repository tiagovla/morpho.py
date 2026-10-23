"""Reproduce the fcc sphere band diagram in Photonic Crystals, Fig. 6.2."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint


def main():
    # The book's conventional cubic lattice constant is normalized to a=1.
    # These are primitive vectors for its fcc lattice.
    a = 1.0
    a1 = (a / 2, a / 2, 0.0)
    a2 = (0.0, a / 2, a / 2)
    a3 = (a / 2, 0.0, a / 2)

    # Coordinates below are in the reciprocal primitive basis (b1, b2, b3).
    X = SPoint((1 / 2, 1 / 2, 0.0), "X")
    U = SPoint((5 / 8, 5 / 8, 1 / 4), "U")
    L = SPoint((1 / 2, 1 / 2, 1 / 2), "L")
    Gamma = SPoint((0.0, 0.0, 0.0), "Γ")
    X2 = SPoint((1 / 2, 1 / 2, 0.0), "X")
    W = SPoint((3 / 4, 1 / 2, 1 / 4), "W")
    K = SPoint((3 / 4, 3 / 8, 3 / 8), "K")
    symmetry_points = [X, U, L, Gamma, X2, W, K]

    points_per_segment = 20
    path = BZPath(
        a1,
        a2,
        a3,
        symmetry_points,
        points_per_segment=points_per_segment,
    )

    # Close-packed spheres have diameter equal to the fcc nearest-neighbor
    # distance a/sqrt(2).  They have epsilon=13 and are embedded in air.
    n_grid = 49
    pwe_order = 7
    radius = a / (2 * np.sqrt(2))
    eps_sphere = 13.0
    geometry = Geometry(a1, a2, a3, n_grid, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y, z):
        # Include periodic images because the sphere intersects primitive-cell
        # boundaries.  The 27 neighboring cells contain the nearest images.
        distance_squared = np.full(eps_r.shape, np.inf)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    center_x = i * a1[0] + j * a2[0] + k * a3[0]
                    center_y = i * a1[1] + j * a2[1] + k * a3[1]
                    center_z = i * a1[2] + j * a2[2] + k * a3[2]
                    distance_squared = np.minimum(
                        distance_squared,
                        (x - center_x) ** 2
                        + (y - center_y) ** 2
                        + (z - center_z) ** 2,
                    )
        eps_r[distance_squared <= radius**2] = eps_sphere

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

    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "_static"
        / "book_ch6_fig2_3d_reproduction.png"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print("Structure: fcc close-packed spheres, epsilon=13 in air")
    print(f"Plane-wave cutoff: {pwe_order} x {pwe_order} x {pwe_order}")
    print(f"Sampled {len(path_distance)} points on X-U-L-Γ-X-W-K")
    print("The book reports no complete photonic band gap for this structure.")
    print(f"Saved band diagram to {output}")


if __name__ == "__main__":
    main()
