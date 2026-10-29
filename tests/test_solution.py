"""Check returned solver results against homogeneous-medium dispersion."""

import numpy as np
import pytest

from morpho import BrillouinZonePath, Geometry, Solution, Solver, SymmetryPoint


@pytest.mark.parametrize(
    "dimension, polarization",
    [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)],
)
def test_solver_returns_independent_solution(dimension, polarization):
    vectors = np.eye(dimension)
    points = [
        SymmetryPoint(tuple(np.zeros(dimension)), "Γ"),
        SymmetryPoint(tuple(vectors[0] / 2), "X"),
    ]
    path = BrillouinZonePath(*vectors, points, n_points=3)
    geometry = Geometry(*vectors, *([5] * dimension))
    options = {name: 1 for name in ("P", "Q", "R")[:dimension]}
    if polarization is not None:
        options["pol"] = polarization
    solver = Solver(geometry, path, **options)

    solution = solver.run()

    assert isinstance(solution, Solution)
    assert solution.path is path
    assert solution.pol == polarization
    np.testing.assert_allclose(solution.path_distance, [0.0, np.pi / 2, np.pi])
    band_count = 2 if dimension == 3 else 1
    expected = np.linalg.norm(path.betas.values, axis=0)
    np.testing.assert_allclose(
        solution.wn, np.repeat(expected[:, None], band_count, axis=1), atol=1e-12
    )
    np.testing.assert_allclose(solution.wn, np.stack(solver.wn))
    assert len(solution.modes) == len(solver.modes) == 3
    assert all(mode.shape[1] == band_count for mode in solution.modes)

    geometry.eps_r[:] = 4.0
    next_solution = solver.run()

    np.testing.assert_allclose(next_solution.wn, solution.wn / 2, atol=1e-12)
    np.testing.assert_allclose(solution.wn[:, 0], expected, atol=1e-12)
    assert len(solver.wn) == len(next_solution.modes) == len(solution.modes) == 3
    assert next_solution.modes is not solution.modes


@pytest.mark.parametrize(
    "dimension, polarization", [(1, "TM"), (1, "TE"), (2, "TM"), (2, "TE"), (3, None)]
)
@pytest.mark.parametrize("loss", [0.0, 0.2])
@pytest.mark.parametrize("store_modes", [True, False])
def test_partial_and_frequency_only_match_full_solution(
    dimension, polarization, loss, store_modes
):
    vectors = np.eye(dimension)
    path = BrillouinZonePath(
        *vectors,
        [
            SymmetryPoint(tuple(np.full(dimension, 0.1)), "A"),
            SymmetryPoint(tuple(np.full(dimension, 0.23)), "B"),
        ],
        n_points=2,
    )
    geometry = Geometry(*vectors, *([7] * dimension))
    geometry.eps_r[:] = np.where(geometry.X.x < 0, 4 + 1j * loss, 2 + 0.5j * loss)
    geometry.mu_r[:] = np.where(geometry.X.x < 0, 1.2 + 0.1j * loss, 1)
    options = dict(zip(("P", "Q", "R")[:dimension], [3] * dimension))
    if polarization is not None:
        options["pol"] = polarization
    solver = Solver(geometry, path, **options)
    full = solver.run()
    partial = solver.run(n_bands=2, store_modes=store_modes)
    np.testing.assert_allclose(partial.wn, full.wn[:, :2], atol=1e-8)
    assert len(full.modes) == 2
    if store_modes:
        assert len(partial.modes) == 2
        assert all(mode.shape[1] == 2 for mode in partial.modes)
    else:
        assert partial.modes == solver.modes == []
    frequency_only = solver.run(store_modes=False)
    np.testing.assert_allclose(frequency_only.wn, full.wn, atol=1e-8)
    assert frequency_only.modes == []


@pytest.mark.parametrize("n_bands", [0, -1, 1.5, True, 4])
def test_reject_invalid_requested_band_counts(n_bands):
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    with pytest.raises(ValueError, match="n_bands"):
        Solver(Geometry((1.0,), 7), path, P=3).run(n_bands=n_bands)


def test_reject_nonboolean_store_modes():
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    with pytest.raises(ValueError, match="store_modes"):
        Solver(Geometry((1.0,), 7), path).run(store_modes=1)


@pytest.mark.parametrize("n_bands", [1, 2, 3])
def test_partial_3d_keeps_physical_gamma_zero_modes(n_bands):
    vectors = np.eye(3)
    path = BrillouinZonePath(
        *vectors,
        [SymmetryPoint((0.0, 0.0, 0.0), "Γ"), SymmetryPoint((0.1, 0.1, 0.1), "K")],
        n_points=2,
    )
    result = Solver(Geometry(*vectors, 7, 7, 7), path, P=3, Q=3, R=3).run(
        n_bands=n_bands, store_modes=False
    )
    expected = [0.0, 0.0, 2 * np.pi][:n_bands]
    np.testing.assert_allclose(result.wn[0], expected, atol=3e-7)
