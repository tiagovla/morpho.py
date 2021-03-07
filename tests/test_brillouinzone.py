"""Test brillouinzone.py module."""
import numpy as np
import pytest
from morpho.brillouinzone import BrillouinZonePath as BZPath
from morpho.brillouinzone import SymmetryPoint as SPoint


def test_symmetrypoint_constructor_3d():
    X = SPoint((0.4, 0.5, 0.3), "X")
    assert X.name == "X"
    assert X.point.shape == (3, )


def test_symmetrypoint_constructor_2d():
    X = SPoint((0.5, 0.3), "X")
    assert X.name == "X"
    assert X.point.shape == (2, )


def test_symmetrypoint_constructor_1d():
    X = SPoint((0.5, ), "X")
    assert X.name == "X"
    assert X.point.shape == (1, )


def test_brillouin_zone_path_3d():
    a = 1
    n_points = 11
    t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)
    G = SPoint((0, 0, 0), "Γ")
    Z = SPoint((0, 0, 1 / 2), "Z")
    X = SPoint((1 / 2, 0, 0), "X")

    path = BZPath([G, Z, X], t1, t2, t3, n_points=n_points)
    assert path._T1 == pytest.approx(np.array([2 * np.pi / a, 0, 0]))
    assert path._T2 == pytest.approx(np.array([0, 2 * np.pi / a, 0]))
    assert path._T3 == pytest.approx(np.array([0, 0, 2 * np.pi / a]))
    assert path.beta_vec.shape == (3, n_points)


def test_brillouin_zone_path_2d():
    a = 1
    n_points = 11
    t1, t2 = (a, 0), (0, a)
    G = SPoint((0, 0), "Γ")
    X = SPoint((1, 0), "X")
    Y = SPoint((0, 1), "Y")

    path = BZPath([G, X, Y], t1, t2, n_points=n_points)
    assert path._T1 == pytest.approx(np.array([2 * np.pi / a, 0, 0]))
    assert path._T2 == pytest.approx(np.array([0, 2 * np.pi / a, 0]))
    assert path.beta_vec.shape == (2, n_points)


def test_brillouin_zone_path_1d():
    a = 1
    n_points = 11
    t1 = (a, )
    G = SPoint((0, ), "Γ")
    X = SPoint((1, ), "X")

    path = BZPath([G, X], t1, n_points=n_points)
    assert path._T1 == pytest.approx(np.array([2 * np.pi / a, 0, 0]))
    assert path.beta_vec.shape == (1, n_points)
