"""Physical Gamma eigenvectors must satisfy curl and divergence constraints."""

import numpy as np
import pytest

from morpho import BrillouinZonePath, Geometry, Solver, SymmetryPoint
from morpho.utils import convmat, harmonic_indices


@pytest.mark.parametrize("counts", [(1, 1, 1), (3, 3, 3), (2, 3, 2)])
@pytest.mark.parametrize("n_bands", [1, 2, 4])
def test_gamma_static_modes_satisfy_gauss_and_maxwell(counts, n_bands):
    count = np.prod(counts)
    n_bands = min(n_bands, 2 * count)
    vectors = np.array([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, 0.2, 1.1]])
    path = BrillouinZonePath(
        *vectors,
        [SymmetryPoint((0.0, 0.0, 0.0), "Γ"), SymmetryPoint((0.1, 0.07, 0.04), "K")],
        n_points=2,
    )
    geometry = Geometry(*vectors, 7, 7, 7)
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4.0, 2.0)
    geometry.mu_r[:] = np.where(geometry.X.y < 0, 1.5, 1.0)
    solver = Solver(geometry, path, **dict(zip(("P", "Q", "R"), counts)))
    result = solver.run(n_bands=n_bands)
    modes = result.modes[0]
    epsilon = np.kron(np.eye(3), convmat(geometry.eps_r, *counts))
    grids = np.meshgrid(*(harmonic_indices(value) for value in counts), indexing="ij")
    indices = np.stack([grid.ravel(order="F") for grid in grids])
    k = -2 * np.pi * np.linalg.inv(vectors) @ indices
    displacement = (epsilon @ modes).reshape(3, count, -1)
    np.testing.assert_allclose(
        np.sum(k[:, :, None] * displacement, axis=0), 0, atol=1e-11
    )
    static_count = min(2, n_bands)
    electric = modes[:, :static_count].reshape(3, count, -1)
    curl = np.cross(k.T[:, None, :], electric.transpose(1, 2, 0))
    np.testing.assert_allclose(curl, 0, atol=1e-11)
    np.testing.assert_allclose(
        modes.conj().T @ epsilon @ modes, np.eye(n_bands), atol=1e-11
    )
    np.testing.assert_array_equal(result.wn[0, :static_count], 0.0)
    frequency_only = solver.run(n_bands=n_bands, store_modes=False)
    np.testing.assert_allclose(frequency_only.wn, result.wn, atol=1e-10)
