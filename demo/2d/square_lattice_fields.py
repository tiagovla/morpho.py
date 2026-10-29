"""Electric and magnetic mode intensities for a square lattice of rods."""

import matplotlib.pyplot as plt

from morpho import (
    BlochPoints,
    Geometry,
    Solver,
    plot_field,
    reconstruct_fields,
)


def main():
    a1, a2 = (1.0, 0.0), (0.0, 1.0)
    sampling = BlochPoints(a1, a2, points=[(0.5, 0.0)])

    def eps_rf(eps_r, x, y):
        eps_r[x**2 + y**2 <= 0.2**2] = 8.9

    geometry = Geometry(a1, a2, 65, 65, eps_rf=eps_rf)
    solution = Solver(geometry, sampling, P=7, Q=7, pol="TM").run(n_bands=1)
    fields = reconstruct_fields(solution)

    _, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
    plot_field(fields, field="E", ax=axes[0], cmap="magma")
    plot_field(fields, field="H", ax=axes[1], cmap="magma")
    axes[0].set_title("TM at X: electric field |E|²")
    axes[1].set_title("TM at X: magnetic field |Z₀H|²")
    plt.show()


if __name__ == "__main__":
    main()
