# API reference

## Brillouin-zone paths

::: morpho.brillouinzone.SymmetryPoint

::: morpho.brillouinzone.BlochVectors

::: morpho.brillouinzone.BrillouinZonePath

::: morpho.brillouinzone.BrillouinZonePath1D

::: morpho.brillouinzone.BrillouinZonePath2D

::: morpho.brillouinzone.BrillouinZonePath3D

## Solutions

::: morpho.solver.Solution

Every solver's ``run()`` method returns a ``Solution`` containing its path,
eigen-wavenumbers, modes, and polarization. Reusing a solver starts a fresh
calculation; previously returned results keep their calculated data.
``solution.path_distance`` returns the cumulative wave-vector distance at each
sample. ``solution.wave_vectors`` contains the Cartesian sampling coordinates
with shape `(dimension, number_of_samples)`. Both arrays are read-only snapshots.
The result owns a copy of the input path, including its lattice, labels, and
symmetry points, so later edits to the original path do not change existing
results or their plots. Consequently, `solution.path is path` is false.

## Plotting

::: morpho.plotting.plot_brillouin_zone

::: morpho.plotting.plot_bands

Pass the returned result directly to ``plot_bands``. It retrieves the path from
the solution, scales frequencies by ``|a1|/(2π)``, and shades gaps common to all
supplied solutions along the sampled path. Use ``lattice_constant=a`` when the
conventional reference period differs from ``|a1|``. Both plotting functions
return the Matplotlib axes. Keep both functions separate and pass the same
parent axes to put the Brillouin zone inside the band plot:

```python
import matplotlib.pyplot as plt

from morpho import plot_bands, plot_brillouin_zone

solution = solver.run()
_, band_axis = plt.subplots(figsize=(7, 5), tight_layout=True)
plot_bands(solution, ax=band_axis)
zone_axis = plot_brillouin_zone(
    solution.path, ax=band_axis, inset=True, inset_bounds=(0.32, 0.12, 0.30, 0.25)
)
```

The zone inset has a transparent background, so underlying bands remain visible.
Its axes are returned separately for further styling. Omitting
``inset=True`` draws a standalone Brillouin zone in the supplied axes.
Use ``inset_bounds=(left, bottom, width, height)`` to place it in clear space.
For 3D zones, rotate the returned axes with ``zone_axis.view_init(elev=25, azim=-105)``.

Pass multiple polarizations together with
``plot_bands((tm_solution, te_solution), ax=band_axis)`` to shade their shared gaps.

## Band gaps

::: morpho.bands.BandGap

::: morpho.bands.find_band_gaps

Use the same gap calculation without opening a figure:

```python
from morpho import find_band_gaps

for gap in find_band_gaps((tm_solution, te_solution), n_bands=4):
    print(gap.lower, gap.upper, 100 * gap.gap_to_midgap_ratio)
```

Edges use `omega a / (2 pi c)`, with `a = |a1|` unless `lattice_constant=a`
is supplied. Ratios are fractions: `0.314` means `31.4%`. Multiple solutions
return only their shared gaps. `n_bands` limits the bands analyzed, so gaps
above that limit are not reported. Touching edges and gaps within numerical
tolerance are excluded. Results apply only to the sampled path, not the whole
Brillouin zone. Complex-frequency spectra are rejected; use the real and
imaginary frequency plots to inspect lossy results instead.

## Geometry

::: morpho.geometry.Geometry

::: morpho.geometry.GeometryBase

::: morpho.geometry.Geometry1D

::: morpho.geometry.Geometry2D

::: morpho.geometry.Geometry3D

## Solvers

::: morpho.solver.Solver

::: morpho.solver.Solver1D

::: morpho.solver.Solver2D

::: morpho.solver.Solver3D

`solver.run(n_bands=4, store_modes=False)` computes only the four lowest bands
for real, positive materials and skips eigenvector calculation and storage.
`solution.modes` and `solver.modes` are empty in this mode. The defaults still
calculate all physical bands and retain their modes. `n_bands` must be a positive
integer no greater than the retained physical band count (`P`, `P * Q`, or
`2 * P * Q * R`). For general complex eigenproblems, band selection follows a
full eigenvalue solve; frequency-only runs still skip eigenvectors. In 3D the
longitudinal null modes are excluded from the returned physical band count.

Solvers reject invalid polarization labels (use exactly `"TM"` or `"TE"`),
nonpositive or noninteger harmonic counts, and incompatible geometry/path
lattices. Each material-grid dimension needs at least `2 * harmonics - 1`
samples to represent every Fourier difference used by the convolution matrix.
Paths need at least two finite points and, with global sampling, an integer
`n_points >= 2`. Material samples must be finite.

### Lossy materials

All dimensions accept frequency-independent, scalar complex `eps_r` and `mu_r`.
Material samples must be finite and nonzero. Real, positive materials use a
Hermitian eigensolver; complex or nonpositive materials use a general eigensolver.
`Solution.wn` contains complex `omega / c`, ordered by real part and then
imaginary part, with the principal square-root convention. For ordinary passive
dielectrics, use positive imaginary material parts with `exp(-i omega t)`;
negative imaginary frequencies describe temporal decay at real Bloch vectors.
This does not solve for complex Bloch vectors at a fixed real frequency.
Frequency dispersion, anisotropic tensors, and nonlinear material models are
not included. Negative material constants are treated as nondispersive inputs;
physical metal models generally require frequency dispersion.

For example, set `geometry.eps_r[:] = 4 + 0.2j` without a material callback,
then call `solver.run()`. Use `plot_bands(solution)` for real frequencies or
`plot_bands(solution, component="imag")` for imaginary frequencies. Lossy bands
are never shaded as conventional band gaps.

## Fourier utilities

::: morpho.utils.harmonic_indices

::: morpho.utils.convmat
