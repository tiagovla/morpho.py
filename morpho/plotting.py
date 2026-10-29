"""Matplotlib helpers for Brillouin zones and band structures."""

from numbers import Integral
from .fields import Fields

import numpy as np
from numpy.linalg import norm

from .bands import _find_band_gaps, _prepare_bands
from .sampling import BlochPoints


def plot_brillouin_zone(
    path, ax=None, *, inset=False, inset_bounds=(0.65, 0.06, 0.32, 0.32), **kwargs
):
    """Plot the first Brillouin zone and symmetry path in 2D or 3D.

    Parameters
    ----------
    path : BrillouinZonePath2D, BrillouinZonePath3D, or BlochPoints
        Sampling whose reciprocal lattice vectors define the zone. Unordered
        Bloch points are shown as a scatter plot without connecting lines.
    ax : matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Axes to draw on, or the parent band axes when ``inset=True``.
        Create suitable axes when omitted.
    inset : bool, default=False
        Place the zone inside ``ax`` with a transparent background. Requires
        parent axes; 3D paths create a 3D inset inside the 2D band plot.
    inset_bounds : tuple of float, default=(0.65, 0.06, 0.32, 0.32)
        Inset left, bottom, width, and height as fractions of the parent axes.
        Used only when ``inset=True``. Rotate returned 3D axes with ``view_init``.
    **kwargs
        Polygon style options passed to Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes containing the Brillouin zone, or the newly created inset axes.
    """
    if path.dim not in (2, 3):
        raise ValueError("Only 2D and 3D Brillouin zones can be plotted.")
    if inset and ax is None:
        raise ValueError("Parent axes are required for a Brillouin-zone inset.")

    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    reciprocal_basis = np.column_stack(
        [getattr(path, f"b{index}") for index in range(1, path.dim + 1)]
    )
    unit_indices = np.vstack(
        (np.eye(path.dim, dtype=int), -np.eye(path.dim, dtype=int))
    )
    vectors = unit_indices @ reciprocal_basis.T
    halfspaces = np.column_stack(
        (2 * vectors, -np.einsum("ij,ij->i", vectors, vectors))
    )
    vertices = HalfspaceIntersection(halfspaces, np.zeros(path.dim)).intersections

    radius = norm(vertices, axis=1).max()
    min_singular_value = np.linalg.svd(reciprocal_basis, compute_uv=False).min()
    index_limit = int(np.ceil(2 * radius / min_singular_value))
    # Reciprocal vectors longer than twice this radius cannot cut the cell.
    index_ranges = [range(-index_limit, index_limit + 1)] * path.dim
    indices = np.stack(np.meshgrid(*index_ranges, indexing="ij"), axis=-1).reshape(
        -1, path.dim
    )
    vectors = indices @ reciprocal_basis.T
    vectors = vectors[np.any(indices != 0, axis=1)]
    vectors = vectors[norm(vectors, axis=1) <= 2 * radius]
    halfspaces = np.column_stack(
        (2 * vectors, -np.einsum("ij,ij->i", vectors, vectors))
    )
    vertices = HalfspaceIntersection(halfspaces, np.zeros(path.dim)).intersections

    style = {
        "facecolor": "none" if inset else "tab:blue",
        "edgecolor": "0.35" if inset else "black",
        "alpha": 1.0 if inset else 0.2,
        "linewidth": 0.7 if inset else 1.0,
    }
    style.update(kwargs)
    if inset:
        projection = {"projection": "3d"} if path.dim == 3 else {}
        ax = ax.inset_axes(inset_bounds, **projection)
        ax.patch.set_alpha(0.0)
        ax.set_in_layout(False)
    elif ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"} if path.dim == 3 else None)

    if path.dim == 2:
        vertices = vertices[ConvexHull(vertices).vertices]
        ax.fill(vertices[:, 0], vertices[:, 1], **style)
        ax.axhline(0.0, xmin=0.1, xmax=0.9, color="0.55", lw=0.6, zorder=0)
        ax.axvline(0.0, ymin=0.1, ymax=0.9, color="0.55", lw=0.6, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    else:
        from matplotlib.text import Annotation
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from mpl_toolkits.mplot3d.proj3d import proj_transform

        class PathLabel3D(Annotation):
            """Keep labels outside the path cluster as the 3D view rotates."""

            def __init__(self, name, point, center, **font):
                self.point = point
                self.center = center
                super().__init__(
                    name,
                    xy=(0, 0),
                    xytext=(0, 0),
                    textcoords="offset points",
                    annotation_clip=False,
                    zorder=5,
                    **font,
                )

            def draw(self, renderer):
                projection = self.axes.get_proj()
                self.xy = proj_transform(*self.point, projection)[:2]
                center = proj_transform(*self.center, projection)[:2]
                direction = np.asarray(self.xy) - center
                distance = norm(direction)
                direction = direction / distance if distance else np.array([-1.0, -1.0])
                self.set_position(5 * direction)
                self.set_ha("left" if direction[0] >= 0 else "right")
                self.set_va("bottom" if direction[1] >= 0 else "top")
                super().draw(renderer)

        hull = ConvexHull(vertices)
        faces = []
        planes = []
        for equation in hull.equations:
            if any(
                np.linalg.norm(equation[:3] - plane[:3]) < 1e-8
                and abs(equation[3] - plane[3]) < 1e-8 * radius
                for plane in planes
            ):
                continue
            planes.append(equation)
            face = vertices[
                np.abs(vertices @ equation[:3] + equation[3]) < 1e-8 * radius
            ]
            center = face.mean(axis=0)
            u = face[0] - center
            u /= norm(u)
            v = np.cross(equation[:3], u)
            angles = np.arctan2((face - center) @ v, (face - center) @ u)
            faces.append(face[np.argsort(angles)])

        ax.add_collection3d(Poly3DCollection(faces, **style))
        ax.auto_scale_xyz(*vertices.T)
        ax.set_box_aspect(np.ptp(vertices, axis=0))
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_zlabel(r"$k_z$")

    if isinstance(path, BlochPoints):
        ax.scatter(*path.betas.values, color="black", s=3 if inset else 8)
    elif path.path_points:
        points = (
            np.stack([point.point for point in path.path_points]) @ reciprocal_basis.T
        )
        if path.dim == 2 and len(points) > 2 and np.allclose(points[0], points[-1]):
            ax.fill(points[:, 0], points[:, 1], color="#79c9d9", alpha=0.75)
        ax.plot(
            *points.T,
            color="black",
            marker="o",
            markersize=2.5 if inset else 4,
            lw=0.9 if inset else 1.5,
        )
        seen = set()
        label_center = np.unique(points, axis=0).mean(axis=0)
        for point, name in zip(points, path.point_names):
            key = (tuple(point), name)
            if key in seen:
                continue
            seen.add(key)
            font = {"fontsize": 8 if inset else 10, "fontfamily": "serif"}
            if path.dim == 2:
                if np.allclose(point, 0.0):
                    offset = (-8, -12)
                else:
                    offset = (4, -10 if np.isclose(point[1], 0.0) else 4)
                ax.annotate(
                    name, point, xytext=offset, textcoords="offset points", **font
                )
            else:
                ax.add_artist(PathLabel3D(name, point, label_center, **font))

    if path.dim == 2:
        ax.margins(0.2)
    if inset:
        ax.set_axis_off()
    return ax


def plot_bands(
    solution,
    n_bands=None,
    ax=None,
    *,
    lattice_constant=None,
    component="real",
    **kwargs,
):
    """Plot solved bands and any gaps that remain open along the path.

    Parameters
    ----------
    solution : Solution or sequence of Solution
        Results returned by ``Solver.run()``. Multiple solutions must use the
        same path; only gaps shared by all supplied solutions are shaded.
        Unordered Bloch samples have no path distance and cannot be drawn as
        band paths; use ``find_band_gaps`` to analyze their spectra.
    n_bands : int, optional
        Number of lowest bands to plot. Plot all bands when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Create axes when omitted.
    lattice_constant : float, optional
        Reference length ``a`` for normalized frequency ``omega a / (2 pi c)``.
        Default to the length of the first direct lattice vector. Supply the
        conventional period when it differs, such as for an fcc primitive cell.
    component : {"real", "imag"}, default="real"
        Frequency component to draw. Lossy results have no gap shading.
    **kwargs
        Line style options passed to Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the band plot. Use ``ax.figure`` to access its figure.
    """
    import matplotlib.pyplot as plt

    if component not in ("real", "imag"):
        raise ValueError("component must be 'real' or 'imag'.")
    solutions, band_sets = _prepare_bands(solution, n_bands, lattice_constant)
    path = solutions[0].path
    path_distance = solutions[0].path_distance
    point_locations = path.point_locations
    point_names = path.point_names
    lossy = any(np.any(bands.imag) for bands in band_sets)
    gaps = [] if lossy or component == "imag" else _find_band_gaps(band_sets, n_bands)
    if ax is None:
        _, ax = plt.subplots()

    for gap in gaps:
        ax.axhspan(gap.lower, gap.upper, color="#ffe600", alpha=0.8, zorder=0)

    labels = []
    for index, (result, bands) in enumerate(zip(solutions, band_sets)):
        style = dict(kwargs)
        label = style.pop("label", result.pol)
        if "color" not in style and "c" not in style:
            style["color"] = f"C{index}"
        frequencies = (bands.real if component == "real" else bands.imag)[:, :n_bands]
        lines = ax.plot(path_distance, frequencies, **style)
        if label:
            lines[0].set_label(label)
            labels.append(label)
    ax.set_xticks(point_locations)
    ax.set_xticklabels(point_names)
    ax.set_xlim(0.0, point_locations[-1])
    ax.set_xlabel(r"Bloch wave vector $k$")
    if component == "imag":
        ax.set_ylabel(
            r"Imaginary normalized frequency $\mathrm{Im}(\omega a / (2\pi c))$"
        )
    elif lossy:
        ax.set_ylabel(r"Real normalized frequency $\mathrm{Re}(\omega a / (2\pi c))$")
    else:
        ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.grid(True, alpha=0.25)
    if labels:
        ax.legend(frameon=False)
    return ax


def plot_field(
    fields,
    *,
    field="E",
    component="intensity",
    part="real",
    ax=None,
    slice_axis=2,
    slice_index=None,
    **kwargs,
):
    """Plot a field profile, a 2D map, or a planar slice through a 3D cell.

    Parameters
    ----------
    fields : Fields
        Output of ``reconstruct_fields``.
    field : {"E", "H"}, default="E"
        Electric field or impedance-scaled magnetic field.
    component : {"intensity", "x", "y", "z"}, default="intensity"
        Squared vector magnitude, or a Cartesian component.
    part : {"real", "imag", "abs"}, default="real"
        Component representation; ignored for intensity maps.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; create axes when omitted.
    slice_axis : int, default=2
        Lattice-grid direction held fixed for a 3D slice.
    slice_index : int, optional
        Slice sample index; default to the middle grid sample.
    **kwargs
        Styling passed to ``plot`` in 1D or ``pcolormesh`` in 2D and 3D.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the field profile or map. Maps include a colorbar.
    """
    if not isinstance(fields, Fields):
        raise TypeError("Pass the Fields returned by reconstruct_fields().")
    if (
        field not in ("E", "H")
        or component not in ("intensity", "x", "y", "z")
        or part not in ("real", "imag", "abs")
    ):
        raise ValueError(
            "Choose field E/H, component intensity/x/y/z, and part real/imag/abs."
        )
    values = getattr(fields, field)
    dimension = values.ndim - 1
    if component == "intensity":
        values = np.sum(np.abs(values) ** 2, axis=-1)
        label = f"|{field}|²"
    else:
        values = values[..., ("x", "y", "z").index(component)]
        values = np.abs(values) if part == "abs" else getattr(values, part)
        label = f"{part}({field}_{component})"
    positions = tuple(vars(fields.coordinates).values())
    labels = ("x", "y")
    if dimension == 3:
        if (
            not isinstance(slice_axis, Integral)
            or isinstance(slice_axis, bool)
            or not 0 <= slice_axis < 3
        ):
            raise ValueError("slice_axis must be 0, 1, or 2.")
        if slice_index is None:
            slice_index = values.shape[slice_axis] // 2
        if (
            not isinstance(slice_index, Integral)
            or isinstance(slice_index, bool)
            or not 0 <= slice_index < values.shape[slice_axis]
        ):
            raise ValueError("slice_index must select an existing grid sample.")
        values = np.take(values, slice_index, axis=slice_axis)
        positions = np.stack(
            [np.take(position, slice_index, axis=slice_axis) for position in positions],
            axis=-1,
        )
        first, second = np.delete(fields.lattice_vectors, slice_axis, axis=0)
        first = first / np.linalg.norm(first)
        second = second - np.dot(first, second) * first
        second /= np.linalg.norm(second)
        positions = (positions @ first, positions @ second)
        labels = ("u (distance in slice)", "v (distance in slice)")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    if dimension == 1:
        ax.plot(positions[0], values, **kwargs)
        ax.set_ylabel(label)
    else:
        style = {"shading": "auto", **kwargs}
        artist = ax.pcolormesh(*positions, values, **style)
        ax.figure.colorbar(artist, ax=ax, label=label)
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal")
    ax.set_xlabel(labels[0])
    return ax
