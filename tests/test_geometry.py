"""Material callbacks must not retain previous parameter values."""

import numpy as np
import pytest

from morpho import Geometry


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("material", ["eps_r", "mu_r"])
def test_callback_rebuilds_profile_when_inclusion_shrinks(dimension, material):
    geometry = Geometry(*np.eye(dimension), *([9] * dimension))
    radius = 0.4

    def eps_rf(*, eps_r, x, **coordinates):
        eps_r[np.abs(x) <= radius] = 9.0 + 0.2j

    def mu_rf(values, x, **coordinates):
        values[np.abs(x) <= radius] = 9.0 + 0.2j

    geometry.overwrite(eps_rf if material == "eps_r" else mu_rf)
    before = getattr(geometry, material).copy()
    radius = 0.1
    after = getattr(geometry, material)
    expected = np.where(np.abs(geometry.X.x) <= radius, 9.0 + 0.2j, 1.0)
    np.testing.assert_array_equal(after, expected)
    assert np.count_nonzero(before != 1) > np.count_nonzero(after != 1)
    np.testing.assert_array_equal(getattr(geometry, material), expected)


def test_direct_material_edits_persist_without_callback():
    geometry = Geometry((1.0,), 9)
    geometry.eps_r[:] = 4.0
    geometry.mu_r[:] = 2.0
    np.testing.assert_array_equal(geometry.eps_r, 4.0)
    np.testing.assert_array_equal(geometry.mu_r, 2.0)


def test_callback_failure_preserves_last_successful_profile():
    geometry = Geometry((1.0,), 9)
    geometry.eps_r[:] = 4.0

    @geometry.overwrite
    def eps_rf(eps_r, x):
        eps_r[:] = 9.0
        raise RuntimeError("invalid parameters")

    with pytest.raises(RuntimeError, match="invalid parameters"):
        _ = geometry.eps_r
    np.testing.assert_array_equal(geometry._eps_r, 4.0)
