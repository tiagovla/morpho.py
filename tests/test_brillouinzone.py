"""Test brillouinzone.py module."""
import numpy as np
import pytest
from morpho.brillouinzone import BrillouinZonePath as BZPath
from morpho.brillouinzone import SymmetryPoint as SPoint


def test_symmetrypoint_constructor_3d():
    """Test 3D SymmetryPoint constructor."""
    X = SPoint((0.4, 0.5, 0.3), "X")
    assert X.name == "X"
    assert X.point.shape == (3, )


def test_symmetrypoint_constructor_2d():
    """Test 2D SymmetryPoint constructor."""
    X = SPoint((0.5, 0.3), "X")
    assert X.name == "X"
    assert X.point.shape == (2, )


def test_symmetrypoint_constructor_1d():
    """Test 1D SymmetryPoint constructor."""
    X = SPoint((0.5, ), "X")
    assert X.name == "X"
    assert X.point.shape == (1, )


def test_brillouin_zone_path_3d():
    """Test 3D BrillouinZonePath."""
    a = 1
    n_points = 11
    t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)
    G = SPoint((0, 0, 0), "Γ")
    Z = SPoint((0, 0, 1 / 2), "Z")
    X = SPoint((1 / 2, 0, 0), "X")

    path = BZPath(t1, t2, t3, [G, Z, X], n_points)
    assert path.b1 == pytest.approx(np.array([2 * np.pi / a, 0, 0]))
    assert path.b2 == pytest.approx(np.array([0, 2 * np.pi / a, 0]))
    assert path.b3 == pytest.approx(np.array([0, 0, 2 * np.pi / a]))
    assert path.betas.values.shape == (3, n_points)


def test_brillouin_zone_path_2d():
    """Test 2D BrillouinZonePath."""
    a = 1
    n_points = 11
    t1, t2 = (a, 0), (0, a)
    G = SPoint((0, 0), "Γ")
    X = SPoint((1, 0), "X")
    Y = SPoint((0, 1), "Y")

    path = BZPath(t1, t2, [G, X, Y], n_points=n_points)
    assert path.b1 == pytest.approx(np.array([2 * np.pi / a, 0]))
    assert path.b2 == pytest.approx(np.array([0, 2 * np.pi / a]))
    assert path.betas.values.shape == (2, n_points)


def test_brillouin_zone_path_1d():
    """Test 1D BrillouinZonePath."""
    a = 1
    n_points = 11
    t1 = (a, )
    G = SPoint((0, ), "Γ")
    X = SPoint((1, ), "X")

    path = BZPath(t1, [G, X], n_points=n_points)
    assert path.b1 == pytest.approx(np.array([2 * np.pi / a]))
    assert path.betas.values.shape == (1, n_points)
