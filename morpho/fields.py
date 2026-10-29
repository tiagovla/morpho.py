"""Reconstruct electric and magnetic Bloch fields on the sampled unit cell."""

from dataclasses import dataclass
from numbers import Integral
from typing import Union

import numpy as np

from .geometry import CartesianVectors1D, CartesianVectors2D, CartesianVectors3D
from .solver import Solution
from .utils import convmat, harmonic_indices


@dataclass
class Fields:
    """Complex fields on the material grid, with vector components last.

    ``E`` and ``H`` have shape ``(*grid_shape, 3)``. ``H`` is the
    impedance-scaled magnetic field ``Z0 H``. Eigenmode amplitudes and overall
    phase are arbitrary; squared magnitudes are not power or energy densities.
    ``coordinates`` contains the Cartesian positions of the samples.
    """

    E: np.ndarray
    H: np.ndarray
    coordinates: Union[CartesianVectors1D, CartesianVectors2D, CartesianVectors3D]
    lattice_vectors: np.ndarray

    @property
    def electric_intensity(self) -> np.ndarray:
        """Return ``sum(abs(E)**2)`` over vector components."""
        return np.sum(np.abs(self.E) ** 2, axis=-1)

    @property
    def magnetic_intensity(self) -> np.ndarray:
        """Return ``sum(abs(Z0 H)**2)`` over vector components."""
        return np.sum(np.abs(self.H) ** 2, axis=-1)


def _sample_fourier(coefficients, counts, shape):
    spectrum = np.zeros((*shape, 3), dtype=complex)
    indices = np.ix_(
        *(harmonic_indices(count) % size for count, size in zip(counts, shape))
    )
    spectrum[indices] = coefficients.reshape((*counts, 3), order="F")
    return np.fft.ifftn(spectrum, axes=tuple(range(len(shape)))) * np.prod(shape)


def reconstruct_fields(solution, *, sample=0, band=0) -> Fields:
    """Reconstruct one mode's E and impedance-scaled H on its material grid.

    Parameters
    ----------
    solution : Solution
        Solver result computed with ``store_modes=True``.
    sample : int, default=0
        Zero-based Bloch-vector sample index.
    band : int, default=0
        Zero-based retained band index.

    Returns
    -------
    Fields
        Complex Cartesian fields, including the Bloch phase ``exp(-i beta.r)``
        and using time dependence ``exp(-i omega t)``. The periodic Fourier
        basis is ``exp(+i G.(r-r_origin))``, with ``r_origin`` the grid corner.
        At zero frequency the primary field is reconstructed and its
        undetermined companion field is set to zero.
    """
    if not isinstance(solution, Solution):
        raise TypeError("Pass the Solution returned by Solver.run().")
    if not solution.modes or solution.geometry is None or solution.harmonics is None:
        raise ValueError(
            "Field reconstruction requires a solver result with stored modes."
        )
    for name, value, size in (
        ("sample", sample, solution.wn.shape[0]),
        ("band", band, solution.wn.shape[1]),
    ):
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or not 0 <= value < size
        ):
            raise ValueError(f"{name} must be an integer between 0 and {size - 1}.")
    geometry = solution.geometry
    counts = solution.harmonics
    dimension = len(counts)
    count = np.prod(counts)
    primary = solution.modes[sample][:, band]
    expected_size = 3 * count if dimension == 3 else count
    if primary.shape != (expected_size,) or not np.isfinite(primary).all():
        raise ValueError("Mode coefficients do not match the stored harmonics.")
    grids = np.meshgrid(*(harmonic_indices(value) for value in counts), indexing="ij")
    indices = np.stack([grid.ravel(order="F") for grid in grids])
    vectors = np.stack(
        [getattr(geometry, f"a{index + 1}") for index in range(dimension)]
    )
    reciprocal = 2 * np.pi * np.linalg.inv(vectors)
    beta = solution.wave_vectors[:, sample]
    wave_vectors = np.zeros((count, 3))
    wave_vectors[:, :dimension] = (reciprocal @ indices - beta[:, None]).T
    electric = np.zeros((count, 3), dtype=complex)
    magnetic = np.zeros_like(electric)
    frequency = solution.wn[sample, band]
    if dimension == 3 or solution.pol == "TM":
        if dimension == 3:
            electric[:] = primary.reshape(3, count).T
        else:
            electric[:, 2] = primary
        if frequency != 0:
            inverse_mu = convmat(1 / geometry.mu_r, *counts)
            magnetic = inverse_mu @ np.cross(wave_vectors, electric) / frequency
    elif solution.pol == "TE":
        magnetic[:, 2] = primary
        if frequency != 0:
            inverse_epsilon = (
                np.linalg.inv(convmat(geometry.eps_r, *counts))
                if dimension == 2
                else convmat(1 / geometry.eps_r, *counts)
            )
            electric = -inverse_epsilon @ np.cross(wave_vectors, magnetic) / frequency
    else:
        raise ValueError("Scalar fields require a TM or TE polarization label.")
    coordinates = geometry.X
    positions = tuple(vars(coordinates).values())
    phase = np.exp(
        -1j * sum(value * coordinate for value, coordinate in zip(beta, positions))
    )
    E = _sample_fourier(electric, counts, geometry.eps_r.shape) * phase[..., None]
    H = _sample_fourier(magnetic, counts, geometry.eps_r.shape) * phase[..., None]
    return Fields(E, H, coordinates, vectors)


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
        label = f"|{field}|² (arbitrary units)"
    else:
        values = values[..., ("x", "y", "z").index(component)]
        values = np.abs(values) if part == "abs" else getattr(values, part)
        label = f"{part}({field}_{component}) (arbitrary units)"
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
