"""Reject invalid solver requests before matrix assembly."""

import numpy as np
import pytest

from morpho import BrillouinZonePath, Geometry, Solver, SymmetryPoint
from morpho.utils import convmat, harmonic_indices


def make_problem(dimension=2):
    vectors = np.eye(dimension)
    path = BrillouinZonePath(
        *vectors,
        [
            SymmetryPoint(tuple(np.zeros(dimension)), "Γ"),
            SymmetryPoint(tuple(vectors[0] / 2), "X"),
        ],
        n_points=3,
    )
    return Geometry(*vectors, *([9] * dimension)), path


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "3"])
def test_harmonic_count_requires_positive_integer(count):
    with pytest.raises(ValueError, match="positive integer"):
        harmonic_indices(count)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("polarization", ["tm", "typo", "", None])
def test_invalid_polarization(dimension, polarization):
    geometry, path = make_problem(dimension)
    with pytest.raises(ValueError, match="pol must"):
        Solver(geometry, path, pol=polarization).run()


@pytest.mark.parametrize(
    "shape, counts", [((5,), (4,)), ((5, 7), (3, 5)), ((5, 7, 3), (3, 4, 3))]
)
def test_grid_must_cover_all_harmonic_differences(shape, counts):
    with pytest.raises(ValueError, match="grid dimension"):
        convmat(np.ones(shape), *counts)


@pytest.mark.parametrize("size", [5, 6])
def test_smallest_valid_grid(size):
    np.testing.assert_allclose(convmat(np.ones(size), P=3), np.eye(3))


def test_reject_nonfinite_samples():
    with pytest.raises(ValueError, match="finite"):
        convmat(np.array([1.0, np.nan, 1.0]))


def test_reject_mismatched_geometry_dimension_and_lattice():
    geometry, path = make_problem()
    with pytest.raises(ValueError, match="same dimension"):
        Solver(make_problem(1)[0], path).run()
    geometry.a1 *= 2
    with pytest.raises(ValueError, match="same lattice"):
        Solver(geometry, path).run()


@pytest.mark.parametrize("n_points", [0, 1, 2.5, True])
def test_invalid_path_sample_count(n_points):
    geometry, path = make_problem()
    path.n_points = n_points
    with pytest.raises(ValueError, match="n_points"):
        Solver(geometry, path).run()


def test_reject_singular_lattice_and_invalid_path_points():
    geometry, path = make_problem()
    path.a2[:] = path.a1
    with pytest.raises(ValueError, match="linearly independent"):
        Solver(geometry, path).run()
    geometry, path = make_problem()
    path.path_points[0] = SymmetryPoint((0.0,), "Γ")
    with pytest.raises(ValueError, match="finite points"):
        Solver(geometry, path).run()
