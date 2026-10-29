"""Analyze sampled band gaps without creating a figure."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import List

import numpy as np
from numpy.linalg import norm

from .solver import Solution


@dataclass(frozen=True)
class BandGap:
    """A gap in normalized frequency ``omega a / (2 pi c)``.

    Parameters
    ----------
    lower : float
        Maximum frequency of the lower band on the sampled path.
    upper : float
        Minimum frequency of the upper band on the sampled path.
    """

    lower: float
    upper: float

    @property
    def gap_to_midgap_ratio(self) -> float:
        """Return fractional gap width; multiply by 100 for a percentage."""
        return 2 * (self.upper - self.lower) / (self.upper + self.lower)


def _prepare_bands(solution, n_bands, lattice_constant):
    solutions = list(solution) if isinstance(solution, (list, tuple)) else [solution]
    if not solutions:
        raise ValueError("At least one solution is required.")
    if any(not isinstance(result, Solution) for result in solutions):
        raise TypeError("Pass the Solution returned by Solver.run().")
    if n_bands is not None and (
        not isinstance(n_bands, Integral) or isinstance(n_bands, bool) or n_bands < 1
    ):
        raise ValueError("n_bands must be a positive integer.")

    path = solutions[0].path
    wave_vectors = solutions[0].wave_vectors
    if lattice_constant is None:
        lattice_constant = norm(path.a1)
    if (
        not isinstance(lattice_constant, Real)
        or isinstance(lattice_constant, bool)
        or not np.isfinite(lattice_constant)
        or lattice_constant <= 0
    ):
        raise ValueError("lattice_constant must be a positive finite length.")
    band_sets = []
    for result in solutions:
        if result.path is not path and (
            not np.array_equal(result.wave_vectors, wave_vectors)
            or not np.array_equal(result.path.point_locations, path.point_locations)
            or result.path.point_names != path.point_names
            or norm(result.path.a1) != norm(path.a1)
        ):
            raise ValueError("All solutions must use the same path.")
        if (
            result.wn.ndim != 2
            or result.wn.shape[0] != wave_vectors.shape[1]
            or result.wn.shape[1] == 0
            or not np.isfinite(result.wn).all()
            or np.any(result.wn.real < 0)
            or np.any(np.diff(result.wn.real, axis=1) < 0)
        ):
            raise ValueError(
                "Solution bands must be finite, nonnegative, ordered by real "
                "frequency, with one row per path sample."
            )
        band_sets.append(result.wn * lattice_constant / (2 * np.pi))
    return solutions, band_sets


def _is_open_gap(lower, upper):
    return upper > lower and not np.isclose(upper, lower, rtol=1e-8, atol=1e-12)


def _find_band_gaps(band_sets, n_bands=None):
    if any(np.any(bands.imag) for bands in band_sets):
        raise ValueError(
            "Band gaps require real frequencies; lossy spectra have no conventional band gaps."
        )
    gaps = None
    for bands in band_sets:
        bands = bands.real[:, :n_bands]
        solver_gaps = []
        for index in range(bands.shape[1] - 1):
            lower = bands[:, index].max()
            upper = bands[:, index + 1].min()
            if _is_open_gap(lower, upper):
                solver_gaps.append((lower, upper))
        if gaps is None:
            gaps = solver_gaps
        else:
            gaps = [
                (max(lower, other_lower), min(upper, other_upper))
                for lower, upper in gaps
                for other_lower, other_upper in solver_gaps
                if _is_open_gap(max(lower, other_lower), min(upper, other_upper))
            ]
    return [BandGap(float(lower), float(upper)) for lower, upper in gaps or []]


def find_band_gaps(solution, n_bands=None, *, lattice_constant=None) -> List[BandGap]:
    """Return real-frequency gaps shared by all supplied solutions.

    Parameters
    ----------
    solution : Solution or sequence of Solution
        One result, or multiple polarizations sampled on the same path.
    n_bands : int, optional
        Use the lowest requested number of bands; default to all stored bands.
    lattice_constant : float, optional
        Reference length for ``omega a / (2 pi c)``; default to ``|a1|``.

    Returns
    -------
    list of BandGap
        Gaps between adjacent retained bands, ordered low to high. Each gap
        has normalized ``lower`` and ``upper`` edges and a fractional
        ``gap_to_midgap_ratio``. A gap on a sampled path does not establish a
        complete gap over the Brillouin zone. Lossy spectra are rejected.
    """
    _, band_sets = _prepare_bands(solution, n_bands, lattice_constant)
    return _find_band_gaps(band_sets, n_bands)
