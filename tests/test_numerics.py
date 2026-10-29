"""Independent Fourier, dispersion, and heterogeneous convergence references."""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import j1

from morpho import BrillouinZonePath, Geometry, Solver, SymmetryPoint
from morpho.utils import convmat


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
def test_multiharmonic_dispersion_on_oblique_lattice(dimension, polarization):
    vectors = np.array([[1.0, 0.0, 0.0], [0.3, 0.9, 0.0], [0.1, 0.2, 1.1]])[
        :dimension, :dimension
    ]
    points = [
        SymmetryPoint(tuple(np.zeros(dimension)), "Γ"),
        SymmetryPoint(tuple(np.full(dimension, 0.21)), "K"),
    ]
    path = BrillouinZonePath(*vectors, points, n_points=2)
    geometry = Geometry(*vectors, *([9] * dimension))
    geometry.eps_r[:] = 4
    geometry.mu_r[:] = 2
    counts = [3] * dimension
    options = dict(zip(("P", "Q", "R")[:dimension], counts))
    if polarization is not None:
        options["pol"] = polarization
    result = Solver(geometry, path, **options).run()
    indices = np.stack(np.meshgrid(*([[-1, 0, 1]] * dimension), indexing="ij")).reshape(
        dimension, -1
    )
    reciprocal = 2 * np.pi * np.linalg.inv(vectors) @ indices
    for beta, frequencies in zip(path.betas.values.T, result.wn):
        expected = np.sort(
            np.linalg.norm(beta[:, None] - reciprocal, axis=0) / np.sqrt(8)
        )
        if dimension == 3:
            expected = np.repeat(expected, 2)
        np.testing.assert_allclose(frequencies, expected, atol=2e-7)
    if dimension == 3:
        assert np.count_nonzero(result.wn[0] < 1e-6) == 2


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_bilayer_converges_to_transfer_matrix_dispersion(polarization):
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.23,), "A"), SymmetryPoint((0.37,), "B")], n_points=2
    )
    geometry = Geometry((1.0,), 1024)
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4.0, 1.0)
    expected = []
    for beta in path.betas.values[0]:

        def relation(frequency):
            return (
                np.cos(frequency) * np.cos(frequency / 2)
                - 1.25 * np.sin(frequency) * np.sin(frequency / 2)
                - np.cos(beta)
            )

        intervals = np.linspace(0, 8, 401)
        roots = [
            brentq(relation, left, right)
            for left, right in zip(intervals[:-1], intervals[1:])
            if relation(left) * relation(right) < 0
        ]
        expected.append(roots[:2])
    errors = []
    for count in (3, 25):
        result = Solver(geometry, path, P=count, pol=polarization).run()
        errors.append(np.max(np.abs(result.wn[:, :2] - expected)))
    assert errors[1] < 0.1 * errors[0]
    assert errors[1] < (1e-4 if polarization == "TM" else 0.04)


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_square_rods_converge_to_analytic_circle_fourier_reference(polarization):
    count = 13
    indices = np.arange(-6, 7)
    harmonics = np.stack(
        [grid.ravel(order="F") for grid in np.meshgrid(indices, indices, indexing="ij")]
    )
    differences = harmonics[:, :, None] - harmonics[:, None, :]
    argument = 2 * np.pi * 0.2 * np.linalg.norm(differences, axis=0)
    circle = np.ones_like(argument)
    np.divide(2 * j1(argument), argument, out=circle, where=argument != 0)
    epsilon = np.eye(count**2) + 7.9 * np.pi * 0.2**2 * circle
    inverse_epsilon = np.linalg.inv(epsilon)
    path = BrillouinZonePath(
        (1.0, 0.0),
        (0.0, 1.0),
        [SymmetryPoint((0.5, 0.0), "X"), SymmetryPoint((0.5, 0.5), "M")],
        n_points=2,
    )
    reference = []
    for beta in path.betas.values.T:
        k = beta[:, None] - 2 * np.pi * harmonics
        if polarization == "TM":
            A = np.diag(np.sum(k**2, axis=0))
            B = epsilon
        else:
            A = sum(row[:, None] * inverse_epsilon * row[None, :] for row in k)
            B = None
        reference.append(np.sqrt(eigh(A, B, eigvals_only=True)[:4]) / (2 * np.pi))
    geometry = Geometry((1.0, 0.0), (0.0, 1.0), 257, 257)
    geometry.eps_r[:] = np.where(geometry.X.x**2 + geometry.X.y**2 <= 0.2**2, 8.9, 1.0)
    errors = []
    for count in (5, 13):
        result = Solver(geometry, path, P=count, Q=count, pol=polarization).run()
        errors.append(np.max(np.abs(result.wn[:, :4] / (2 * np.pi) - reference)))
    assert errors[1] < 0.02 * errors[0]
    assert errors[1] < 3e-4


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
def test_heterogeneous_lossy_limit_recovers_lossless_bands(dimension, polarization):
    vectors = np.eye(dimension)
    path = BrillouinZonePath(
        *vectors,
        [
            SymmetryPoint(tuple(np.full(dimension, 0.1)), "A"),
            SymmetryPoint(tuple(np.full(dimension, 0.2)), "B"),
        ],
        n_points=2,
    )
    geometry = Geometry(*vectors, *([9] * dimension))
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4.0, 2.0)
    options = dict(zip(("P", "Q", "R")[:dimension], [3] * dimension))
    if polarization is not None:
        options["pol"] = polarization
    real_result = Solver(geometry, path, **options).run()
    geometry.eps_r[:] += 1e-7j
    complex_result = Solver(geometry, path, **options).run()
    np.testing.assert_allclose(complex_result.wn.real, real_result.wn, atol=1e-8)
    assert np.max(np.abs(complex_result.wn.imag)) < 1e-6


@pytest.mark.parametrize("count", [3, 4])
def test_convolution_matches_known_sinusoidal_fourier_coefficients(count):
    x = np.arange(17) / 17
    samples = 4 + np.cos(2 * np.pi * x) + 0.4 * np.sin(4 * np.pi * x)
    differences = np.arange(count)[:, None] - np.arange(count)[None, :]
    expected = 4 * (differences == 0) + 0.5 * (np.abs(differences) == 1)
    expected = (
        expected.astype(complex)
        - 0.2j * (differences == 2)
        + 0.2j * (differences == -2)
    )
    np.testing.assert_allclose(convmat(samples, count), expected, atol=1e-14)
