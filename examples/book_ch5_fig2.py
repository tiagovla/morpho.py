"""Reproduce the square-rod band diagram in Photonic Crystals, Fig. 5.2."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint
from morpho.brillouinzone import BlochVectors


def sample_path(a1, a2, points, points_per_segment):
    """Sample each symmetry segment including every vertex exactly."""
    reference = BZPath(a1, a2, points, n_points=2)
    vertices = np.stack([point.point for point in points], axis=1)
    vertices = (
        vertices[0] * reference.b1[:, None]
        + vertices[1] * reference.b2[:, None]
    )

    k_values = []
    distances = []
    offset = 0.0
    for start, end in zip(vertices.T[:-1], vertices.T[1:]):
        length = np.linalg.norm(end - start)
        segment = np.linspace(start, end, points_per_segment + 1)
        segment_distance = offset + np.linspace(
            0.0, length, points_per_segment + 1
        )
        if k_values:
            segment = segment[1:]
            segment_distance = segment_distance[1:]
        k_values.extend(segment)
        distances.extend(segment_distance)
        offset += length

    class SampledPath:
        dim = 2

        def __init__(self):
            self.b1 = reference.b1
            self.b2 = reference.b2
            self.betas = BlochVectors(np.asarray(k_values).T, np.asarray(distances))

    point_locations = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(vertices, axis=1), axis=0)))
    )
    return SampledPath(), np.asarray(distances), point_locations


def main():
    # Book, Chapter 5, Figure 2: alumina rods in air on a square lattice.
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

    path, path_distance, point_locations = sample_path(
        a1, a2, symmetry_points, points_per_segment
    )
    geometry = Geometry(a1, a2, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y):
        mask = x**2 + y**2 <= radius**2
        eps_r[mask] = eps_rod

    tm_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TM")
    te_solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, pol="TE")
    tm_solver.run()
    te_solver.run()

    scale = a / (2 * np.pi)
    tm = np.vstack(tm_solver.wn) * scale
    te = np.vstack(te_solver.wn) * scale
    tm_lower_edge = np.max(tm[:, 0])
    tm_upper_edge = np.min(tm[:, 1])
    gap_ratio = (
        200.0
        * (tm_upper_edge - tm_lower_edge)
        / (tm_upper_edge + tm_lower_edge)
    )
    gap_error = abs(gap_ratio - 31.4)

    figure, axis = plt.subplots(figsize=(7, 5))
    for band in range(4):
        axis.plot(path_distance, tm[:, band], color="tab:blue", lw=1.4)
        axis.plot(path_distance, te[:, band], color="tab:red", lw=1.4)
    axis.set_xticks(point_locations)
    axis.set_xticklabels([p.name for p in symmetry_points])
    axis.set_xlim(0.0, path_distance[-1])
    axis.set_ylim(0.0, 0.8)
    axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
    axis.set_xlabel(r"Bloch wave vector $k$")
    axis.text(
        0.02, 0.98, "TM", color="tab:blue", transform=axis.transAxes, va="top"
    )
    axis.text(
        0.12, 0.98, "TE", color="tab:red", transform=axis.transAxes, va="top"
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "_static"
        / "book_ch5_fig2_reproduction.png"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"TM band 1 maximum along Γ-X-M-Γ: {tm_lower_edge:.6f}")
    print(f"TM band 2 minimum along Γ-X-M-Γ: {tm_upper_edge:.6f}")
    print(
        f"TM gap-to-midgap ratio: {gap_ratio:.2f}% "
        f"(book reference: 31.4%, difference: {gap_error:.2f} percentage points)"
    )
    if gap_error > 1.0:
        raise SystemExit("The computed gap differs from the book reference by over 1 pp.")
    print(f"Saved band diagram to {output}")


if __name__ == "__main__":
    main()
