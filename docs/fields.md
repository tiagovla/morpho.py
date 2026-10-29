# Electromagnetic fields

Compute a solution with modes retained, then reconstruct a selected band at a
selected Bloch-vector sample:

```python
from morpho import reconstruct_fields, plot_field

solution = solver.run(n_bands=4)
fields = reconstruct_fields(solution, sample=1, band=0)
plot_field(fields)
plot_field(fields, field="H", component="y", part="real")
```

Sample and band indices start at zero. `fields.E` and `fields.H` are complex
arrays with shape `(*material_grid_shape, 3)`, with Cartesian x, y, z components
on the last axis. `fields.H` contains `Z0 H`, the magnetic field multiplied by
vacuum impedance. `fields.coordinates` contains the Cartesian grid positions.
`fields.electric_intensity` and `fields.magnetic_intensity` give squared vector
magnitudes. Mode amplitude and overall phase are arbitrary; these magnitudes
are not calibrated power or energy densities.

The result stores harmonic counts and an independent material-grid snapshot.
Changing the original geometry, callback parameters, or solver after solving
does not change reconstructed fields. Frequency-only runs retain neither modes
nor material grids, and cannot reconstruct fields.

The spatial convention matches the solver's `beta - G` operators: periodic
Fourier factors are `exp(+i G · (r - r_origin))`, where `r_origin` is the
centered material grid's corner, and the Bloch factor is `exp(-i beta · r)`.
Time dependence is `exp(-i omega t)`. The scalar TM mode is `E_z`; the scalar TE
mode is `Z0 H_z`. The 3D modes are electric-field vectors. Companion fields use
the same material factorization as the solver. At zero frequency, the primary
static field is returned and its undetermined companion field is set to zero.

`plot_field` draws a line in 1D, a Cartesian map in 2D, and a planar slice in
3D. Choose `component="x"`, `"y"`, or `"z"` and `part="real"`, `"imag"`, or
`"abs"`; the default `component="intensity"` draws the squared vector magnitude.

For 3D, `slice_axis` selects the fixed lattice-grid direction and `slice_index`
selects its sample. The default is the middle slice along the third direction.
Axes u and v are orthonormal distances within that plane, so oblique cells are
displayed without treating a lattice coordinate as a Cartesian coordinate.

```python
plot_field(fields, slice_axis=1, slice_index=10, cmap="magma")
```
