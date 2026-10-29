"""Compare reconstructed fields with homogeneous Maxwell plane waves."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from morpho import (
    BrillouinZonePath,
    Geometry,
    Solver,
    SymmetryPoint,
    plot_field,
    reconstruct_fields,
)


def problem(dimension, polarization=None, loss=0.0, counts=None):
    vectors = np.array([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, 0.2, 1.1]])[
        :dimension, :dimension
    ]
    path = BrillouinZonePath(
        *vectors,
        [
            SymmetryPoint(tuple(np.full(dimension, 0.13)), "A"),
            SymmetryPoint(tuple(np.full(dimension, 0.23)), "B"),
        ],
        n_points=2,
    )
    geometry = Geometry(*vectors, *([7] * dimension))
    geometry.eps_r[:] = 4 + 1j * loss
    geometry.mu_r[:] = 1.5 + 0.2j * loss
    counts = counts or (1,) * dimension
    options = dict(zip(("P", "Q", "R"), counts))
    if polarization is not None:
        options["pol"] = polarization
    return geometry, Solver(geometry, path, **options).run()


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
@pytest.mark.parametrize("loss", [0.0, 0.2])
def test_reconstruction_matches_plane_wave_and_maxwell(dimension, polarization, loss):
    geometry, result = problem(dimension, polarization, loss)
    fields = reconstruct_fields(result, sample=1)
    coordinates = tuple(vars(geometry.X).values())
    beta = result.wave_vectors[:, 1]
    phase = np.exp(
        -1j * sum(value * coordinate for value, coordinate in zip(beta, coordinates))
    )
    wave_vector = np.zeros(3)
    wave_vector[:dimension] = -beta
    primary = result.modes[1][:, 0]
    if dimension == 3:
        expected_E = phase[..., None] * primary
    elif polarization == "TM":
        expected_E = np.zeros((*phase.shape, 3), complex)
        expected_E[..., 2] = primary[0] * phase
    else:
        expected_H = np.zeros((*phase.shape, 3), complex)
        expected_H[..., 2] = primary[0] * phase
        expected_E = -np.cross(wave_vector, expected_H) / (
            result.wn[1, 0] * (4 + 1j * loss)
        )
    expected_H = np.cross(wave_vector, expected_E) / (
        result.wn[1, 0] * (1.5 + 0.2j * loss)
    )
    np.testing.assert_allclose(fields.E, expected_E, atol=1e-12)
    np.testing.assert_allclose(fields.H, expected_H, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(wave_vector, fields.H),
        -result.wn[1, 0] * (4 + 1j * loss) * fields.E,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fields.electric_intensity, np.sum(np.abs(fields.E) ** 2, axis=-1)
    )
    np.testing.assert_allclose(
        fields.magnetic_intensity, np.sum(np.abs(fields.H) ** 2, axis=-1)
    )


@pytest.mark.parametrize("counts", [(3,), (2, 3), (3, 2, 2)])
def test_fft_matches_direct_fourier_sum_on_oblique_centered_cell(counts):
    dimension = len(counts)
    geometry, result = problem(
        dimension, "TM" if dimension < 3 else None, counts=counts
    )
    band = result.wn.shape[1] - 1
    fields = reconstruct_fields(result, sample=1, band=band)
    grid_indices = np.meshgrid(
        *(np.arange(size) / size for size in geometry.eps_r.shape), indexing="ij"
    )
    fractional = np.stack([grid.ravel() for grid in grid_indices])
    harmonics = np.meshgrid(
        *(np.arange(-(count // 2), (count + 1) // 2) for count in counts), indexing="ij"
    )
    harmonics = np.stack([grid.ravel(order="F") for grid in harmonics])
    positions = np.stack(
        [coordinate.ravel() for coordinate in vars(geometry.X).values()]
    )
    phase = np.exp(
        1j
        * (2 * np.pi * harmonics.T @ fractional - result.wave_vectors[:, 1] @ positions)
    )
    coefficients = result.modes[1][:, band]
    if dimension == 3:
        expected = (phase.T @ coefficients.reshape(3, -1).T).reshape(fields.E.shape)
    else:
        expected = np.zeros_like(fields.E)
        expected[..., 2] = (coefficients @ phase).reshape(geometry.eps_r.shape)
    np.testing.assert_allclose(fields.E, expected, atol=1e-11)


def test_material_and_harmonic_metadata_survive_input_edits():
    geometry, result = problem(2, "TE")
    expected = reconstruct_fields(result)
    geometry.eps_r[:] = 9.0
    geometry.mu_r[:] = 2.0
    geometry.a1[:] *= 2
    actual = reconstruct_fields(result)
    assert result.harmonics == (1, 1)
    assert result.geometry.eps_rf is result.geometry.mu_rf is None
    np.testing.assert_array_equal(actual.E, expected.E)
    np.testing.assert_array_equal(actual.H, expected.H)
    with pytest.raises(ValueError):
        result.geometry.eps_r[:] = 2.0


def test_static_gamma_field_has_no_undetermined_companion():
    vectors = np.eye(3)
    path = BrillouinZonePath(
        *vectors,
        [SymmetryPoint((0.0, 0.0, 0.0), "Γ"), SymmetryPoint((0.1, 0.1, 0.1), "K")],
        n_points=2,
    )
    geometry = Geometry(*vectors, 7, 7, 7)
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4.0, 2.0)
    result = Solver(geometry, path, P=3, Q=3, R=3).run(n_bands=2)
    fields = reconstruct_fields(result)
    assert np.isfinite(fields.E).all()
    assert fields.electric_intensity.max() > 0
    np.testing.assert_array_equal(fields.H, 0.0)


@pytest.mark.parametrize(
    "options",
    [{"sample": -1}, {"sample": 2}, {"band": 3}, {"band": True}, {"sample": 0.5}],
)
def test_reject_invalid_mode_selection(options):
    _, result = problem(1, "TM")
    with pytest.raises(ValueError):
        reconstruct_fields(result, **options)


def test_frequency_only_results_do_not_retain_material_grids():
    geometry = Geometry((1.0,), 7)
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.1,), "A"), SymmetryPoint((0.2,), "B")], n_points=2
    )
    result = Solver(geometry, path, P=3).run(store_modes=False)
    assert result.geometry is None
    assert result.harmonics == (3,)
    with pytest.raises(ValueError, match="stored modes"):
        reconstruct_fields(result)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_plot_field_profiles_maps_and_slices(dimension):
    _, result = problem(dimension, "TM" if dimension < 3 else None)
    fields = reconstruct_fields(result)
    axis = plot_field(fields)
    if dimension == 1:
        np.testing.assert_allclose(axis.lines[0].get_ydata(), fields.electric_intensity)
    else:
        expected = fields.electric_intensity
        if dimension == 3:
            expected = expected[:, :, expected.shape[2] // 2]
        np.testing.assert_allclose(
            np.asarray(axis.collections[0].get_array()).ravel(), expected.ravel()
        )
        assert len(axis.figure.axes) == 2
    plt.close(axis.figure)
    axis = plot_field(fields, field="H", component="y", part="imag")
    assert axis is not None
    plt.close(axis.figure)


@pytest.mark.parametrize(
    "options",
    [
        {"field": "B"},
        {"component": "w"},
        {"part": "phase"},
        {"slice_axis": 3},
        {"slice_index": -1},
    ],
)
def test_plot_field_rejects_invalid_options(options):
    _, result = problem(3)
    with pytest.raises(ValueError):
        plot_field(reconstruct_fields(result), **options)
