"""Implement photonic crystal geometries."""

from abc import ABC
from collections import namedtuple
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np


class GeometryBase(ABC):
    """Base Geometry class."""

    def __init__(self, eps_rf: Optional[Callable], mu_rf: Optional[Callable]):
        self.eps_rf: Optional[Callable] = eps_rf or None
        self.mu_rf: Optional[Callable] = mu_rf or None
        self._eps_r: np.ndarray
        self._mu_r: np.ndarray

    @property
    def eps_r(self) -> np.ndarray:
        if self.eps_rf:
            self.eps_rf()
        return self._eps_r

    @property
    def mu_r(self) -> np.ndarray:
        if self.mu_rf:
            self.mu_rf()
        return self._mu_r

    def set_eps_rf(self, func: Callable):
        """Set eps_r by a function decorator.

        Parameters
        ----------
        func : Callable
            func
        """
        self.eps_rf = func
        return func

    def set_mu_rf(self, func: Callable):
        """Set mu_r by a function decorator.

        Parameters
        ----------
        func : Callable
            func
        """
        self.mu_rf = func
        return func

    def overwrite(self, func: Callable):
        """Overwrite decorator.

        Parameters
        ----------
        func : Callable
            func
        """
        if func.__name__ in ['eps_rf', 'mu_rf']:
            setattr(self, func.__name__, func)
        else:
            raise Exception("Can only overwrite eps_rf or mu_rf.")


class Geometry1D(GeometryBase):
    """Geometry class.

    Parameters
    ----------
    a1 : Tuple[float, ...]
        Direct a1 vector.
    n1 : int
        Number of divisions in the a1 vector direction.
    eps_rf : Optional[Callable]
        Permittivity matrix eps_r if directly supplied.
    mu_rf : Optional[Callable]
        Permeabillity matrix mu_r if directly supplied.
    """

    def __init__(self,
                 a1: Tuple[float, float],
                 n1: int = 64,
                 eps_rf: Optional[Callable] = None,
                 mu_rf: Optional[Callable] = None):
        super().__init__(eps_rf, mu_rf)
        self.a1: np.ndarray = np.array(a1)
        self.n1: int = n1
        self._eps_r: np.ndarray = np.ones((n1, ), dtype=complex)
        self._mu_r: np.ndarray = np.ones((n1, ), dtype=complex)

    @property
    def X(self) -> NamedTuple:
        X = namedtuple('CartesianVectors1D', ['x'])
        P0 = np.linspace(-0.5, 0.5, self.n1)
        x = P0 * self.a1[0]
        return X(x)


class Geometry2D(GeometryBase):
    """Geometry class.

    Parameters
    ----------
    a1 : Tuple[float, ...]
        Direct a1 vector.
    a2 : Optional[Tuple[float, ...]]
        Direct a2 vector.
    n1 : int
        Number of divisions in the a1 vector direction.
    n2 : int
        Number of divisions in the a2 vector direction.
    eps_rf : Optional[Callable]
        Permittivity matrix eps_r if directly supplied.
    mu_rf : Optional[Callable]
        Permeabillity matrix mu_r if directly supplied.
    """

    def __init__(self,
                 a1: Tuple[float, float],
                 a2: Tuple[float, float],
                 n1: int = 64,
                 n2: int = 64,
                 eps_rf: Optional[Callable] = None,
                 mu_rf: Optional[Callable] = None):
        super().__init__(eps_rf, mu_rf)
        self.a1: np.ndarray = np.array(a1)
        self.a2: np.ndarray = np.array(a2)
        self.n1: int = n1
        self.n2: int = n2
        self._eps_r: np.ndarray = np.ones((n1, n2), dtype=complex)
        self._mu_r: np.ndarray = np.ones((n1, n2), dtype=complex)

    @property
    def X(self) -> NamedTuple:
        X = namedtuple('CartesianVectors2D', ['x', 'y'])

        P0, Q0 = np.meshgrid(
            np.linspace(-0.5, 0.5, self.n1),
            np.linspace(-0.5, 0.5, self.n2),
        )

        x = P0 * self.a1[0] + Q0 * self.a2[0]
        y = P0 * self.a1[1] + Q0 * self.a2[1]

        return X(x, y)


class Geometry3D(GeometryBase):
    """Geometry class.

    Parameters
    ----------
    a1 : Tuple[float, float, float]
        Direct a1 vector.
    a2 : Optional[Tuple[float, float, float]]
        Direct a2 vector.
    a3 : Optional[Tuple[float, float, float]]
        Direct a3 vector.
    n1 : int
        Number of divisions in the a1 vector direction.
    n2 : int
        Number of divisions in the a2 vector direction.
    n3 : int
        Number of divisions in the a3 vector direction.
    eps_rf : Optional[Callable]
        Permittivity matrix function eps_rf if directly supplied.
    mu_rf : Optional[Callable]
        Permeabillity matrix function mu_rf if directly supplied.
    """

    def __init__(self,
                 a1: Tuple[float, float, float],
                 a2: Tuple[float, float, float],
                 a3: Tuple[float, float, float],
                 n1: int = 64,
                 n2: int = 64,
                 n3: int = 64,
                 eps_rf: Optional[Callable] = None,
                 mu_rf: Optional[Callable] = None):
        super().__init__(eps_rf, mu_rf)
        self.a1: np.ndarray = np.array(a1)
        self.a2: np.ndarray = np.array(a2)
        self.a3: np.ndarray = np.array(a3)
        self.n1: int = n1
        self.n2: int = n2
        self.n3: int = n3
        self._eps_r: np.ndarray = np.ones((n1, n2, n3), dtype=complex)
        self._mu_r: np.ndarray = np.ones((n1, n2, n3), dtype=complex)

    @property
    def b1(self) -> np.ndarray:
        return (2 * np.pi * np.cross(self.a2, self.a3) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def b2(self) -> np.ndarray:
        return (2 * np.pi * np.cross(self.a2, self.a3) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def b3(self) -> np.ndarray:
        return (2 * np.pi * np.cross(self.a2, self.a3) /
                np.dot(self.a1, np.cross(self.a2, self.a3)))

    @property
    def X(self) -> NamedTuple:
        X = namedtuple('CartesianVectors3D', ['x', 'y', 'z'])

        P0, Q0, R0 = np.meshgrid(
            np.linspace(-0.5, 0.5, self.n1),
            np.linspace(-0.5, 0.5, self.n2),
            np.linspace(-0.5, 0.5, self.n3),
        )

        x = P0 * self.a1[0] + Q0 * self.a2[0] + R0 * self.a3[0]
        y = P0 * self.a1[1] + Q0 * self.a2[1] + R0 * self.a3[1]
        z = P0 * self.a1[2] + Q0 * self.a2[2] + R0 * self.a3[2]

        return X(x, y, z)


def Geometry(*args, **kwargs):
    """Geometry factory."""
    dim_obj = {1: Geometry1D, 2: Geometry2D, 3: Geometry3D}
    return dim_obj[len(args[0])](*args, **kwargs)
