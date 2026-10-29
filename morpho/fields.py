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
