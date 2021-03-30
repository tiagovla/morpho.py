"""Implement photonic crystal geometries."""

from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Geometry:
    """Geometry class.

    Parameters
    ----------
    t1 : Tuple[float, ...]
        Direct t1 vector.
    t2 : Optional[Tuple[float, ...]]
        Direct t2 vector.
    t3 : Optional[Tuple[float, ...]]
        Direct t3 vector.
    N1 : int
        Number of divisions in the t1 vector direction.
    N2 : int
        Number of divisions in the t2 vector direction.
    N3 : int
        Number of divisions in the t3 vector direction.
    eps_r : Optional[np.ndarray]
        Permittivity matrix eps_r if directly supplied.
    mu_r : Optional[np.ndarray]
        Permeabillity matrix mu_r if directly supplied.
    """

    def __init__(
        self,
        t1: Tuple[float, ...],
        t2: Optional[Tuple[float, ...]] = None,
        t3: Optional[Tuple[float, ...]] = None,
        N1: int = 32,
        N2: int = 1,
        N3: int = 1,
        eps_r: Optional[np.ndarray] = None,
        mu_r: Optional[np.ndarray] = None,
    ):
        """Initialize the geometry obj."""
        self.dim = len(t1)

        self._t1 = np.pad(np.array(t1), (0, 3 - len(t1)), "constant")
        self._t2 = (np.pad(np.array(t2), (0, 3 - len(t2)), "constant")
                    if t2 else np.array([0, 1, 0]))
        self._t3 = (np.pad(np.array(t3), (0, 3 - len(t1)), "constant")
                    if t3 else np.array([0, 0, 1]))

        self._T1 = (2 * np.pi * np.cross(self._t2, self._t3) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))
        self._T2 = (2 * np.pi * np.cross(self._t3, self._t1) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))
        self._T3 = (2 * np.pi * np.cross(self._t1, self._t2) /
                    np.dot(self._t1, np.cross(self._t2, self._t3)))

        self.eps_r = eps_r or np.ones((N1, N2, N3), dtype=complex)
        self.mu_r = mu_r or np.ones((N1, N2, N3), dtype=complex)

        P0, Q0, R0 = np.meshgrid(
            np.linspace(-0.5, 0.5, N1),
            np.linspace(-0.5, 0.5, N2) if N2 > 1 else 0,
            np.linspace(-0.5, 0.5, N3) if N3 > 1 else 0,
        )

        self.x = P0 * self._t1[0] + Q0 * self._t2[0] + R0 * self._t3[0]
        self.y = P0 * self._t1[1] + Q0 * self._t2[1] + R0 * self._t3[1]
        self.z = P0 * self._t1[2] + Q0 * self._t2[2] + R0 * self._t3[2]

        self.eps_rf: Optional[Callable] = None
        self.mu_rf: Optional[Callable] = None

    def setup(self):
        """Run the epsr_f and mur_f if declared by the set decorators."""
        if self.eps_rf:
            self.eps_rf()
        if self.mu_rf:
            self.mu_rf()

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
