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
sample, equivalent to ``solution.path.betas.cumsum``.

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

## Fourier utilities

::: morpho.utils.harmonic_indices

::: morpho.utils.convmat
