# morpho.py

`morpho.py` computes photonic crystal band structures with the plane-wave
expansion (PWE) method. It supports one-, two-, and three-dimensional periodic
geometries, with scalar TE/TM solvers in 1D and 2D and a vector Maxwell solver
in 3D.

Start with the [quick start](#quick-start) below, browse the
[API reference](api.md), or explore the [examples](examples.md).

## Install

Requires Python 3.9 or newer. Install with pip:

```console
pip install morpho.py
```

??? note "Install in a uv project"

    ```console
    uv add morpho.py
    ```

## Source code

Browse the [morpho.py codebase on GitHub](https://github.com/tiagovla/morpho.py).

## Quick start

This example calculates the first four TM and TE bands of alumina rods in air
on a square lattice. The example uses normalized units with lattice constant
`a = 1`; the plotted frequency is $\omega a/(2\pi c)$.

```python
import matplotlib.pyplot as plt

from morpho import (
    BrillouinZonePath,
    Geometry,
    Solver,
    SymmetryPoint,
    plot_bands,
    plot_brillouin_zone,
)

a = 1.0
radius = 0.2 * a
eps_rod = 8.9

a1, a2 = (a, 0.0), (0.0, a)
gamma = SymmetryPoint((0.0, 0.0), "Γ")
x_point = SymmetryPoint((0.5, 0.0), "X")
m_point = SymmetryPoint((0.5, 0.5), "M")
path = BrillouinZonePath(
    a1, a2, [gamma, x_point, m_point, gamma], points_per_segment=40
)

def eps_rf(eps_r, x, y):
    eps_r[x**2 + y**2 <= radius**2] = eps_rod

geometry = Geometry(a1, a2, 65, 65, eps_rf=eps_rf)

tm = Solver(geometry, path, P=7, Q=7, pol="TM")
te = Solver(geometry, path, P=7, Q=7, pol="TE")
tm_solution = tm.run()
te_solution = te.run()

_, axis = plt.subplots(figsize=(7, 5), tight_layout=True)
plot_bands((tm_solution, te_solution), n_bands=4, ax=axis)
plot_brillouin_zone(
    tm_solution.path, ax=axis, inset=True, inset_bounds=(0.32, 0.12, 0.30, 0.25)
)
plt.show()
```

`Geometry` starts with relative permittivity and permeability equal to one. The
`eps_rf` constructor argument supplies the callback that overwrites values inside
the rods. Every callback evaluation starts from ones, so changing a radius also
clears the previous inclusion.
Callbacks must set any other background explicitly. With no callback, edits to
`geometry.eps_r` and `geometry.mu_r` persist between accesses.
`Solver.run()` returns
a `Solution` containing its path, raw $\omega/c$ values in `solution.wn`,
eigenvectors in `solution.modes`, and polarization in `solution.pol`. The
existing `solver.wn` and `solver.modes` attributes remain available.
`solution.path_distance` provides the cumulative distance of each path sample.

`plot_bands` uses the stored path, normalizes frequencies by $|a_1|/(2\pi)$,
and shades gaps shared by the supplied solutions. Gaps are determined over the
sampled path; this alone does not establish a gap throughout the Brillouin zone.
For a conventional period different from $|a_1|$, pass `lattice_constant=a`;
the fcc examples use this to retain their conventional cubic normalization.

The two plotting functions draw into the same parent axes. With `inset=True`,
`plot_brillouin_zone` creates a transparent inset showing the zone and symmetry
path, using a 3D inset for 3D lattices.

The path uses `points_per_segment`, so each symmetry point is sampled exactly.
For a single total number of samples across the whole path, pass `n_points`
instead.
