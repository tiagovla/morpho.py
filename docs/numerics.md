# Numerical accuracy

Three independent resolutions control a band calculation:

- The real-space material grid resolves inclusion boundaries and Fourier coefficients.
- The retained harmonic counts `P`, `Q`, and `R` truncate the plane-wave basis.
- The Bloch-vector samples resolve extrema along the selected path.

Increase each resolution separately while holding the other two fixed. Compare
the bands and gap edges you need, rather than the highest retained band. Require
changes smaller than the accuracy needed for the calculation. Use absolute
frequency errors near zero; relative errors are not meaningful at Gamma.

For a fixed 2D `geometry` and `path`, inspect harmonic convergence with:

```python
import numpy as np
from morpho import Solver

previous = None
for count in (5, 9, 13):
    solution = Solver(geometry, path, P=count, Q=count, pol="TE").run()
    bands = solution.wn[:, :4] * np.linalg.norm(path.a1) / (2 * np.pi)
    if previous is not None:
        print(count, np.max(np.abs(bands - previous)))
    previous = bands
```

Harmonic counts are numbers of retained terms, not maximum harmonic indices.
Odd counts give a symmetric integer range. Each grid dimension must contain at
least `2 * count - 1` samples, but sharp interfaces usually require more.
Increasing harmonics cannot repair a poorly resolved material boundary.

Useful accuracy references include transfer-matrix dispersion for layered
media, analytic Fourier coefficients for circular inclusions, and homogeneous
dispersion on oblique lattices. Agreement with one reference does not establish
convergence for a different geometry.

Band gaps inferred from a symmetry path apply only to that sampled path.
Establishing a complete gap requires checking the relevant Brillouin-zone
interior as well. Complex-frequency lossy spectra do not define conventional
real-frequency band gaps.
