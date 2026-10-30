# Examples

Band frequencies are normalized as $\omega a/(2\pi c)$. Shaded intervals mark
gaps shared by the displayed polarizations along the sampled symmetry path.
The Brillouin-zone inset shows the lattice's reciprocal cell and the path.
The fcc examples use the conventional cubic period for $a$.

## Electromagnetic mode fields

These maps show electric and impedance-scaled magnetic intensities for the
first TM band at X in the square lattice of alumina rods.

![Electric and magnetic mode intensities](assets/square_lattice_fields.png)

??? info "View source"

    ```python
    --8<-- "demo/2d/square_lattice_fields.py"
    ```

See [electromagnetic fields](fields.md) for reconstruction and normalization.

## Two-dimensional square lattice of dielectric rods

This example models alumina rods in air on a square lattice. The lattice
constant is normalized to $a=1$, the rod radius is $0.2a$, and the rod relative
permittivity is 8.9. The real-space dielectric grid has $129\times129$ samples.
The plane-wave expansion uses orders $P=Q=9$ and samples the symmetry path
$\Gamma$–X–M–$\Gamma$ with 100 points per segment. The plot shows the first four
TM bands in blue and TE bands in red. Its calculated TM gap-to-midgap ratio is
31.45%, compared with 31.4% in the book.

The structure is from Chapter 5, Figure 2 of
[*Photonic Crystals: Molding the Flow of Light*](references.md).

![TM and TE bands for alumina rods on a square lattice](assets/square_lattice_dielectric_rods.png)

??? info "View source"

    ```python
    --8<-- "demo/2d/square_lattice_dielectric_rods.py"
    ```

## Two-dimensional triangular lattice of air columns

This example places air columns in a dielectric substrate with relative
permittivity 13. The triangular lattice constant is normalized to $a=1$, and
the column radius is $0.48a$. The real-space grid has $129\times129$ samples;
the plane-wave expansion uses orders $P=Q=9$. The symmetry path is
$\Gamma$–M–K–$\Gamma$, sampled at 100 points per segment. The yellow region is
the gap shared by the TM and TE bands along the sampled path. The plot shows
the first six bands for each polarization. The calculated gap-to-midgap ratio
is 18.82%, close to the book's 18.6% result.

This structure is in Chapter 5, Figure 10 (page 76) of
[*Photonic Crystals: Molding the Flow of Light*](references.md).

![TM and TE bands for air columns on a triangular lattice](assets/triangular_lattice_air_columns.png)

??? info "View source"

    ```python
    --8<-- "demo/2d/triangular_lattice_air_columns.py"
    ```

## Three-dimensional simple-cubic dielectric rod network

The unit cell has normalized lattice constant $a=1$. Dielectric rods of width
$0.2a$ and relative permittivity 2.34 run along the three cubic axes in an air
background. The dielectric grid has $64^3$ samples, and the plane-wave
expansion uses orders $P=Q=R=3$. The band path is D–Z–$\Gamma$–Z–T–X, with 40
points per segment.

![Bands for a simple-cubic dielectric rod network](assets/simple_cubic_rod_network.png)

??? info "View source"

    ```python
    --8<-- "demo/3d/simple_cubic_rod_network.py"
    ```

## Three-dimensional fcc close-packed dielectric spheres

This example places close-packed spheres with relative permittivity 13 in air.
With the cubic supercell lattice constant normalized to $a=1$, each sphere has
radius $a/(2\sqrt{2})$, so its diameter equals the fcc nearest-neighbor
distance. The real-space grid has $49^3$ samples; the plane-wave expansion uses
orders $P=Q=R=7$. It samples X–U–L–$\Gamma$–X–W–K with 20 points per segment and
plots up to 64 bands. The book reports no complete photonic band gap for this
structure.

This structure is in Chapter 6, Figure 2 of
[*Photonic Crystals: Molding the Flow of Light*](references.md).

![Bands for close-packed dielectric spheres on an fcc lattice](assets/fcc_close_packed_spheres.png)

??? info "View source"

    ```python
    --8<-- "demo/3d/fcc_close_packed_spheres.py"
    ```

## Three-dimensional diamond lattice of air spheres

This example places overlapping air spheres in a dielectric background with
relative permittivity 13. The diamond lattice is represented by two
interpenetrating fcc sublattices, with basis positions $(0,0,0)$ and
$(a/4,a/4,a/4)$. The sphere radius is $0.325a$, the real-space grid has $49^3$
samples, and the plane-wave expansion uses orders $P=Q=R=7$. The path is
X–U–L–$\Gamma$–X–W–K with 20 points per segment. The yellow region marks the gap
between bands two and three; along this sampled path, the calculated
gap-to-midgap ratio is 28.96%, compared with 29.6% in the book.

This structure is in Chapter 6, Figure 3 of
[*Photonic Crystals: Molding the Flow of Light*](references.md).

![Bands and the second-to-third-band gap for air spheres on a diamond lattice](assets/diamond_air_spheres.png)

??? info "View source"

    ```python
    --8<-- "demo/3d/diamond_air_spheres.py"
    ```
