![morpho.py banner](docs/assets/banner.svg)

[![PyPI version](https://img.shields.io/pypi/v/morpho.py)](https://pypi.org/project/morpho.py/)
[![Python 3.9-3.15](https://img.shields.io/badge/Python-3.9--3.15-3776AB?logo=python&logoColor=white)](https://pypi.org/project/morpho.py/)
[![License](https://img.shields.io/github/license/tiagovla/morpho.py)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://tiagovla.github.io/morpho.py)

`morpho.py` computes photonic crystal band structures with the plane-wave
expansion (PWE) method.

## Installation

The package supports Python 3.9 through 3.15. Add it to a `uv` project with:

```console
uv add morpho.py
```

## Quick start: 2D square lattice

This example calculates the first four TM and TE bands for alumina rods in air
on a square lattice. The lattice constant is normalized to `a = 1`.

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

<p align="center">
  <img src="https://tiagovla.github.io/morpho.py/assets/square_lattice_dielectric_rods.png" alt="TM and TE band structure for alumina rods on a square lattice" width="720">
</p>

This reproduces the square-lattice example in Chapter 5, Figure 2 of
[*Photonic Crystals: Molding the Flow of Light*](docs/references.md). Its TM
gap-to-midgap ratio is about 31.45%, compared with 31.4% in the book.

See the [examples documentation](https://tiagovla.github.io/morpho.py/examples/)
for the other demos, their parameters, figures, and comparisons with the book
where applicable.
