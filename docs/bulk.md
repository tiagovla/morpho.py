# Brillouin-zone sampling

A symmetry path samples selected lines through reciprocal space. To estimate
gaps across the Brillouin zone, use an unordered batch of Bloch vectors or a
regular reciprocal-cell mesh with the same `Solver` API.

## Arbitrary Bloch vectors

```python
from morpho import BlochPoints, Solver

sampling = BlochPoints(
    a1, a2,
    points=[(0.0, 0.0), (0.23, 0.17), (0.4, -0.1)],
)
solution = Solver(geometry, sampling, P=7, Q=7).run(n_bands=4)
```

Coordinates are fractional coefficients of the reciprocal lattice vectors by
default. For Cartesian wave vectors in inverse length units, pass
`fractional=False`. A batch may contain a single point and preserves the supplied
ordering and duplicates. Use `solution.wave_vectors` for its Cartesian sampling
coordinates. Modes from a point batch support `reconstruct_fields` exactly as
modes from a symmetry path.

## Uniform meshes

This square-lattice example estimates the first TM gap over a reciprocal cell:

```python
from morpho import Geometry, Solver, brillouin_zone_mesh, find_band_gaps

a1, a2 = (1.0, 0.0), (0.0, 1.0)

def eps_rf(eps_r, x, y):
    eps_r[x**2 + y**2 <= 0.2**2] = 8.9

geometry = Geometry(a1, a2, 65, 65, eps_rf=eps_rf)
mesh = brillouin_zone_mesh(a1, a2, shape=(11, 11))
solution = Solver(geometry, mesh, P=7, Q=7).run(
    n_bands=4, store_modes=False
)
gaps = find_band_gaps(solution)
```

Use one lattice vector in 1D, two in 2D, or three in 3D. An integer `shape`
applies the same count to every direction; a tuple selects separate counts.
Meshes always contain Gamma and omit duplicated periodic endpoints. Their
equal-weight points cover one primitive reciprocal cell. For oblique lattices,
this cell differs from the Wigner-Seitz polyhedron but covers the Brillouin zone
modulo reciprocal translations.

`mesh.grid_shape` records the mesh dimensions. Samples flatten in C order, so
`solution.wn.reshape((*mesh.grid_shape, -1))` restores the grid layout. Increase
mesh density, material-grid resolution, and harmonic counts separately when
checking convergence. A gap found on a finite mesh is an estimate, not a proof
that no unsampled wave vector closes it. Extend or refine sampling near the
candidate band extrema when stronger evidence is needed.

`find_band_gaps((tm_solution, te_solution))` reports gaps shared by both
polarizations when their sampling matches. Lossy spectra have no conventional
real-frequency gaps. Unordered samples have no cumulative path distance;
`plot_bands` is reserved for symmetry paths. `plot_brillouin_zone(mesh)` displays
the samples as points without connecting lines in 2D and 3D.

## Density of states

For a uniform mesh and real frequencies, a histogram estimates the density of
states of the retained bands per unit cell:

```python
import numpy as np

frequencies = solution.wn * np.linalg.norm(a1) / (2 * np.pi)
counts, edges = np.histogram(frequencies.ravel(), bins=40)
density = counts / (solution.wn.shape[0] * np.diff(edges))
```

The integral of this histogram equals the number of retained bands. Its
frequency resolution depends on both the mesh density and the histogram bins;
bands omitted from the solve do not contribute.
