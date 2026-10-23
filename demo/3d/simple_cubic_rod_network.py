"""Band structure of a simple-cubic dielectric rod network (3D)."""
import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint


def main():
    a = 1.0
    rod_width = 0.2 * a
    eps_rod = 2.34
    n_grid = 64
    pwe_order = 3

    a1, a2, a3 = (a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)
    Gamma = SPoint((0.0, 0.0, 0.0), "Γ")
    Z = SPoint((0.0, 0.0, 0.5), "Z")
    X = SPoint((0.5, 0.0, 0.0), "X")
    D = SPoint((0.5, 0.5, 0.5), "D")
    T = SPoint((0.5, 0.0, 0.5), "T")
    path = BZPath(a1, a2, a3, [D, Z, Gamma, Z, T, X], points_per_segment=40)

    geometry = Geometry(a1, a2, a3, n_grid, n_grid, n_grid)

    @geometry.overwrite
    def eps_rf(eps_r, x, y, z):
        edge = a / 2 - rod_width / 2
        rod_x = (np.abs(y) >= edge) & (np.abs(z) >= edge)
        rod_y = (np.abs(x) >= edge) & (np.abs(z) >= edge)
        rod_z = (np.abs(x) >= edge) & (np.abs(y) >= edge)
        eps_r[rod_x | rod_y | rod_z] = eps_rod

    solver = Solver(geometry, path, P=pwe_order, Q=pwe_order, R=pwe_order)
    solver.run()
    frequencies = np.vstack(solver.wn) * a / (2 * np.pi)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(path.betas.cumsum, frequencies, color="black", lw=0.8)
    axis.set_xticks(path.point_locations)
    axis.set_xticklabels(path.point_names)
    axis.set_xlim(0.0, path.point_locations[-1])
    axis.set_ylim(0.0, 1.6)
    axis.set_xlabel(r"Bloch wave vector $k$")
    axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
    axis.grid(True)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
