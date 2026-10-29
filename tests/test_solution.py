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
