"""Matplotlib helpers for Brillouin zones and band structures."""

from numbers import Integral, Real

import numpy as np
from numpy.linalg import norm

from .solver import Solution


def plot_brillouin_zone(
    path, ax=None, *, inset=False, inset_bounds=(0.65, 0.06, 0.32, 0.32), **kwargs
):
    """Plot the first Brillouin zone and symmetry path in 2D or 3D.

    Parameters
    ----------
    path : BrillouinZonePath2D or BrillouinZonePath3D
        Path whose reciprocal lattice vectors define the zone.
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

    if path.path_points:
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
    **kwargs,
):
    """Plot solved bands and any gaps that remain open along the path.

    Parameters
    ----------
    solution : Solution or sequence of Solution
        Results returned by ``Solver.run()``. Multiple solutions must use the
        same path; only gaps shared by all supplied solutions are shaded.
    n_bands : int, optional
        Number of lowest bands to plot. Plot all bands when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Create axes when omitted.
    lattice_constant : float, optional
        Reference length ``a`` for normalized frequency ``omega a / (2 pi c)``.
        Default to the length of the first direct lattice vector. Supply the
        conventional period when it differs, such as for an fcc primitive cell.
    **kwargs
        Line style options passed to Matplotlib.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the band plot. Use ``ax.figure`` to access its figure.
    """
    import matplotlib.pyplot as plt

    solutions = solution if isinstance(solution, (list, tuple)) else [solution]
    if not solutions:
        raise ValueError("At least one solution is required.")
    if any(not isinstance(result, Solution) for result in solutions):
        raise TypeError("Pass the Solution returned by Solver.run().")
    if n_bands is not None and (
        not isinstance(n_bands, Integral) or isinstance(n_bands, bool) or n_bands < 1
    ):
        raise ValueError("n_bands must be a positive integer.")

    path = solutions[0].path
    path_distance = solutions[0].path_distance
    point_locations = path.point_locations
    point_names = path.point_names
    if lattice_constant is None:
        lattice_constant = norm(path.a1)
    if (
        not isinstance(lattice_constant, Real)
        or not np.isfinite(lattice_constant)
        or lattice_constant <= 0
    ):
        raise ValueError("lattice_constant must be a positive finite length.")
    frequency_scale = lattice_constant / (2 * np.pi)
    band_sets = []
    gaps = None
    for result in solutions:
        if result.path is not path and (
            not np.array_equal(result.path.betas.values, path.betas.values)
            or not np.array_equal(result.path.point_locations, point_locations)
            or result.path.point_names != point_names
            or norm(result.path.a1) != norm(path.a1)
        ):
            raise ValueError("All solutions must use the same path.")
        if (
            result.wn.ndim != 2
            or result.wn.shape[0] != len(path_distance)
            or result.wn.shape[1] == 0
            or not np.isfinite(result.wn).all()
        ):
            raise ValueError(
                "Solution bands must be finite with one row per path sample."
            )
        bands = result.wn[:, :n_bands] * frequency_scale
        band_sets.append(bands)

        solver_gaps = []
        for index in range(bands.shape[1] - 1):
            lower = bands[:, index].max()
            upper = bands[:, index + 1].min()
            if upper > lower and not np.isclose(upper, lower, rtol=1e-8, atol=1e-12):
                solver_gaps.append((lower, upper))
        if gaps is None:
            gaps = solver_gaps
        else:
            gaps = [
                (max(lower, other_lower), min(upper, other_upper))
                for lower, upper in gaps
                for other_lower, other_upper in solver_gaps
                if min(upper, other_upper) > max(lower, other_lower)
                and not np.isclose(
                    min(upper, other_upper),
                    max(lower, other_lower),
                    rtol=1e-8,
                    atol=1e-12,
                )
            ]

    if ax is None:
        _, ax = plt.subplots()

    for lower, upper in gaps or []:
        ax.axhspan(lower, upper, color="#ffe600", alpha=0.8, zorder=0)

    labels = []
    for index, (result, bands) in enumerate(zip(solutions, band_sets)):
        style = dict(kwargs)
        label = style.pop("label", result.pol)
        if "color" not in style and "c" not in style:
            style["color"] = f"C{index}"
        lines = ax.plot(path_distance, bands, **style)
        if label:
            lines[0].set_label(label)
            labels.append(label)
    ax.set_xticks(point_locations)
    ax.set_xticklabels(point_names)
    ax.set_xlim(0.0, point_locations[-1])
    ax.set_xlabel(r"Bloch wave vector $k$")
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.grid(True, alpha=0.25)
    if labels:
        ax.legend(frameon=False)
    return ax
