# Quick start

This example calculates the first four TM and TE bands of alumina rods in air
on a square lattice. The example uses normalized units with lattice constant
`a = 1`; the plotted frequency is $\omega a/(2\pi c)$.

```python
import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath, Geometry, Solver, SymmetryPoint

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
geometry = Geometry(a1, a2, 65, 65)

@geometry.overwrite
def eps_rf(eps_r, x, y):
    eps_r[x**2 + y**2 <= radius**2] = eps_rod

tm = Solver(geometry, path, P=7, Q=7, pol="TM")
te = Solver(geometry, path, P=7, Q=7, pol="TE")
tm.run()
te.run()

scale = a / (2 * np.pi)
frequency_tm = np.vstack(tm.wn) * scale
frequency_te = np.vstack(te.wn) * scale

figure, axis = plt.subplots()
for band in range(4):
    axis.plot(path.betas.cumsum, frequency_tm[:, band], "b-")
    axis.plot(path.betas.cumsum, frequency_te[:, band], "r-")
axis.set_xticks(path.point_locations)
axis.set_xticklabels(path.point_names)
axis.set_ylabel(r"Frequency $\omega a / 2\pi c$")
axis.set_xlabel(r"Bloch wave vector $k$")
axis.grid(True)
figure.tight_layout()
plt.show()
```

`Geometry` starts with relative permittivity and permeability equal to one. The
`eps_rf` callback overwrites the values inside the rods. `Solver.run()` stores
the calculated $\omega/c$ values in `solver.wn`; multiply by $a/(2\pi)$ to get
the normalized frequencies plotted above.

The path uses `points_per_segment`, so each symmetry point is sampled exactly.
For a single total number of samples across the whole path, pass `n_points`
instead.
