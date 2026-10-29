"""Complex-frequency dispersion and full Maxwell residuals."""

import numpy as np
import pytest

from morpho import BrillouinZonePath, Geometry, Solver, SymmetryPoint
from morpho.utils import convmat, harmonic_indices


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
@pytest.mark.parametrize(
    "epsilon, permeability",
    [(4 + 0.4j, 1), (4, 1.5 + 0.2j), (1 + 10j, 1 + 10j), (-4 + 0.4j, 1)],
)
def test_complex_homogeneous_dispersion(dimension, polarization, epsilon, permeability):
    vectors = np.eye(dimension)
    path = BrillouinZonePath(
        *vectors,
        [
            SymmetryPoint(tuple(np.zeros(dimension)), "Γ"),
            SymmetryPoint(tuple(vectors[0] * 0.23), "K"),
        ],
        n_points=3,
    )
    geometry = Geometry(*vectors, *([7] * dimension))
    geometry.eps_r[:] = epsilon
    geometry.mu_r[:] = permeability
    options = dict(zip(("P", "Q", "R")[:dimension], [3] * dimension))
    if polarization is not None:
        options["pol"] = polarization
    result = Solver(geometry, path, **options).run()
    indices = np.stack(np.meshgrid(*([[-1, 0, 1]] * dimension), indexing="ij")).reshape(
        dimension, -1
    )
    for sample, beta in enumerate(path.betas.values.T):
        squared = np.sum((beta[:, None] - 2 * np.pi * indices) ** 2, axis=0)
        expected = np.sqrt(squared / complex(epsilon * permeability))
        expected = expected[np.lexsort((expected.imag, expected.real))]
        if dimension == 3:
            expected = np.repeat(expected, 2)
        np.testing.assert_allclose(result.wn[sample], expected, atol=1e-6)
    assert np.iscomplexobj(result.wn)
    assert np.all(result.wn.imag <= 1e-6)


def test_heterogeneous_lossy_3d_modes_satisfy_maxwell_and_divergence():
    vectors = np.eye(3)
    path = BrillouinZonePath(
        *vectors,
        [SymmetryPoint((0.1, 0.07, 0.04), "A"), SymmetryPoint((0.2, 0.09, 0.03), "B")],
        n_points=2,
    )
    geometry = Geometry(*vectors, 7, 7, 7)
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4 + 0.4j, 2 + 0.1j)
    geometry.mu_r[:] = np.where(geometry.X.y < 0, 1 + 0.05j, 1.5 + 0.1j)
    result = Solver(geometry, path, P=3, Q=3, R=3).run()
    epsilon = np.kron(np.eye(3), convmat(geometry.eps_r, 3, 3, 3))
    inverse_mu = np.kron(np.eye(3), convmat(1 / geometry.mu_r, 3, 3, 3))
    grids = np.meshgrid(*([harmonic_indices(3)] * 3), indexing="ij")
    reciprocal = 2 * np.pi * np.stack([grid.ravel(order="F") for grid in grids])
    for beta, frequencies, modes in zip(path.betas.values.T, result.wn, result.modes):
        k = beta[:, None] - reciprocal
        x, y, z = (np.diag(row) for row in k)
        zero = np.zeros_like(x)
        curl = np.block([[zero, -z, y], [z, zero, -x], [-y, x, zero]])
        left = -curl @ inverse_mu @ curl @ modes
        right = (epsilon @ modes) * frequencies[None, :] ** 2
        assert np.linalg.norm(left - right) / np.linalg.norm(left) < 1e-10
        displacement = (epsilon @ modes).reshape(3, 27, -1)
        divergence = np.sum(k[:, :, None] * displacement, axis=0)
        assert np.linalg.norm(divergence) / np.linalg.norm(displacement) < 1e-10


@pytest.mark.parametrize("material", ["eps_r", "mu_r"])
@pytest.mark.parametrize("value", [0, np.nan, np.inf])
def test_reject_zero_and_nonfinite_materials(material, value):
    geometry = Geometry((1.0,), 5)
    getattr(geometry, material)[:] = value
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    with pytest.raises(ValueError, match=material):
        Solver(geometry, path).run()
