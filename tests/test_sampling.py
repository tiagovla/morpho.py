"""Bulk sampling must preserve coordinates, dispersion, and gap estimates."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from morpho import (
    BlochPoints,
    BrillouinZonePath,
    Geometry,
    Solver,
    SymmetryPoint,
    brillouin_zone_mesh,
    find_band_gaps,
    plot_bands,
    plot_brillouin_zone,
    reconstruct_fields,
)


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
@pytest.mark.parametrize("loss", [0.0, 0.2])
def test_arbitrary_points_match_ordered_path_spectra(dimension, polarization, loss):
    vectors = np.array([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, 0.2, 1.1]])[
        :dimension, :dimension
    ]
    points = np.array([[0.0, 0.0, 0.0], [0.23, 0.1, 0.07], [0.1, 0.28, 0.13]])[
        :, :dimension
    ]
    sampling = BlochPoints(*vectors, points=points)
    path = BrillouinZonePath(
        *vectors,
        [SymmetryPoint(tuple(point), str(index)) for index, point in enumerate(points)],
        points_per_segment=1,
    )
    geometry = Geometry(*vectors, *([7] * dimension))
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4 + 1j * loss, 2 + 0.5j * loss)
    options = dict(zip(("P", "Q", "R")[:dimension], [3] * dimension))
    if polarization is not None:
        options["pol"] = polarization
    expected = Solver(geometry, path, **options).run(n_bands=3, store_modes=False)
    actual = Solver(geometry, sampling, **options).run(n_bands=3, store_modes=False)
    np.testing.assert_allclose(actual.wave_vectors, expected.wave_vectors, atol=1e-14)
    np.testing.assert_allclose(actual.wn[1:], expected.wn[1:], atol=1e-10)
    np.testing.assert_allclose(actual.wn[0, 1:], expected.wn[0, 1:], atol=1e-10)
    # The square root magnifies eigenvalue roundoff at Gamma.
    np.testing.assert_allclose(actual.wn[0, 0], expected.wn[0, 0], atol=2e-7)


def test_cartesian_input_and_source_array_snapshot():
    a1, a2 = (1.0, 0.0), (0.3, 0.9)
    points = np.array([[0.0, 0.0], [0.2, -0.1], [0.2, -0.1]])
    reciprocal = 2 * np.pi * np.linalg.inv(np.array([a1, a2]))
    expected = reciprocal @ points.T
    fractional = BlochPoints(a1, a2, points=points)
    cartesian = BlochPoints(a1, a2, points=expected.T, fractional=False)
    points[:] = 8
    np.testing.assert_allclose(fractional.betas.values, expected)
    np.testing.assert_allclose(cartesian.betas.values, expected)
    assert cartesian.betas.values.shape == (2, 3)
    with pytest.raises(ValueError):
        fractional.betas.values[0, 0] = 1


@pytest.mark.parametrize("shape", [(5,), (4, 5), (3, 4, 3)])
@pytest.mark.parametrize("loss", [0.0, 0.2])
def test_uniform_mesh_dispersion_and_periodic_uniqueness(shape, loss):
    dimension = len(shape)
    vectors = np.eye(dimension)
    mesh = brillouin_zone_mesh(*vectors, shape=shape)
    assert mesh.grid_shape == shape
    assert mesh.betas.values.shape == (dimension, np.prod(shape))
    assert np.count_nonzero(np.all(mesh.betas.values == 0, axis=0)) == 1
    fractional = mesh.betas.values.T / (2 * np.pi)
    assert len(np.unique(np.round(fractional % 1, 12), axis=0)) == np.prod(shape)
    np.testing.assert_allclose(
        fractional.reshape((*shape, dimension))[tuple(count // 2 for count in shape)], 0
    )
    geometry = Geometry(*vectors, *([5] * dimension))
    geometry.eps_r[:] = 4 + 1j * loss
    result = Solver(geometry, mesh).run(store_modes=False)
    expected = np.linalg.norm(mesh.betas.values, axis=0) / np.sqrt(4 + 1j * loss)
    bands = 2 if dimension == 3 else 1
    np.testing.assert_allclose(
        result.wn, np.repeat(expected[:, None], bands, axis=1), atol=1e-12
    )
    with pytest.raises(ValueError, match="no path distance"):
        _ = result.path_distance
    with pytest.raises(ValueError, match="no path distance"):
        plot_bands(result)


def test_mesh_integer_shape_and_gap_refinement():
    vectors = np.eye(2)
    geometry = Geometry(*vectors, 33, 33)
    geometry.eps_r[:] = np.where(geometry.X.x**2 + geometry.X.y**2 <= 0.2**2, 8.9, 1.0)
    gaps = []
    for count in (5, 10):
        mesh = brillouin_zone_mesh(*vectors, shape=count)
        assert mesh.grid_shape == (count, count)
        result = Solver(geometry, mesh, P=5, Q=5).run(n_bands=2, store_modes=False)
        gap = find_band_gaps(result)[0]
        assert gap.upper > gap.lower > 0
        gaps.append(gap)
        if count == 5:
            te = Solver(geometry, mesh, P=5, Q=5, pol="TE").run(
                n_bands=2, store_modes=False
            )
            assert find_band_gaps((result, te)) == []
    assert gaps[1].lower >= gaps[0].lower - 1e-12
    assert gaps[1].upper <= gaps[0].upper + 1e-12


def test_single_bloch_point_supports_fields_and_survives_source_changes():
    geometry = Geometry((1.0,), 9)
    sampling = BlochPoints((1.0,), points=[0.2])
    result = Solver(geometry, sampling, P=3).run(n_bands=2)
    fields = reconstruct_fields(result)
    assert fields.E.shape == (9, 3)
    sampling._betas = BlochPoints((1.0,), points=[0.0, 0.1]).betas
    assert result.wave_vectors.shape == (1, 1)
    np.testing.assert_array_equal(reconstruct_fields(result).E, fields.E)


@pytest.mark.parametrize("dimension", [2, 3])
def test_zone_plot_accepts_unordered_mesh(dimension):
    mesh = brillouin_zone_mesh(*np.eye(dimension), shape=3)
    axis = plot_brillouin_zone(mesh)
    assert axis.collections
    if dimension == 2:
        assert len(axis.collections[0].get_offsets()) == 9
    plt.close(axis.figure)


@pytest.mark.parametrize("points", [[], [[0.0, 0.0]], [[np.inf]], [[1j]], [[[0.0]]]])
def test_reject_invalid_point_arrays(points):
    with pytest.raises(ValueError):
        BlochPoints((1.0,), points=points)


@pytest.mark.parametrize(
    "vectors",
    [[], [(0.0,)], [(1.0, 0.0), (2.0, 0.0)], [(1j,)], [(np.nan,)], [("bad",)]],
)
def test_reject_invalid_lattices(vectors):
    with pytest.raises(ValueError):
        BlochPoints(*vectors, points=[[0.0]])


@pytest.mark.parametrize("shape", [0, -1, True, (2,), (2, 0), (2, 1.5)])
def test_reject_invalid_mesh_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        brillouin_zone_mesh(*np.eye(2), shape=shape)


def test_reject_nonboolean_coordinate_flag_and_mismatched_bulk_samples():
    with pytest.raises(ValueError, match="fractional"):
        BlochPoints((1.0,), points=[0.0], fractional="yes")
    geometry = Geometry((1.0,), 7)
    first = Solver(geometry, BlochPoints((1.0,), points=[0.0, 0.1]), P=3).run()
    second = Solver(geometry, BlochPoints((1.0,), points=[0.0, 0.2]), P=3).run()
    with pytest.raises(ValueError, match="same path"):
        find_band_gaps((first, second))
