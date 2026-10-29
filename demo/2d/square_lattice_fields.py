"""Electric and magnetic mode intensities for a square lattice of rods."""

import matplotlib.pyplot as plt

from morpho import (
    BrillouinZonePath,
    Geometry,
    Solver,
    SymmetryPoint,
    plot_field,
    reconstruct_fields,
)


def main():
    a1, a2 = (1.0, 0.0), (0.0, 1.0)
    path = BrillouinZonePath(
        a1,
        a2,
        [SymmetryPoint((0.0, 0.0), "Γ"), SymmetryPoint((0.5, 0.0), "X")],
        n_points=2,
    )

    def eps_rf(eps_r, x, y):
        eps_r[x**2 + y**2 <= 0.2**2] = 8.9

    geometry = Geometry(a1, a2, 65, 65, eps_rf=eps_rf)
    _, axes = plt.subplots(2, 2, figsize=(9, 8), tight_layout=True)
    for row, polarization in enumerate(("TM", "TE")):
        solution = Solver(geometry, path, P=7, Q=7, pol=polarization).run(n_bands=2)
        fields = reconstruct_fields(solution, sample=1, band=0)
        for column, field in enumerate(("E", "H")):
            plot_field(fields, field=field, ax=axes[row, column], cmap="magma")
            axes[row, column].set_title(f"{polarization}, first band at X, |{field}|²")
    plt.show()


if __name__ == "__main__":
    main()
