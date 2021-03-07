"""Implement photonic crystal geometries."""

from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Geometry:
    """Geometry class."""

    def __init__(
        self,
        t1: Tuple[float, ...],
        t2: Optional[Tuple[float, ...]] = None,
        t3: Optional[Tuple[float, ...]] = None,
        Nx: int = 32,
        Ny: int = 1,
        Nz: int = 1,
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

        self.eps_r = eps_r or np.ones((Nx, Ny, Nz), dtype=complex)
        self.mu_r = mu_r or np.ones((Nx, Ny, Nz), dtype=complex)

        P0, Q0, R0 = np.meshgrid(
            np.linspace(-0.5, 0.5, Nx),
            np.linspace(-0.5, 0.5, Ny) if Ny > 1 else 0,
            np.linspace(-0.5, 0.5, Nz) if Nz > 1 else 0,
        )

        self.x = P0 * self._t1[0] + Q0 * self._t2[0] + R0 * self._t3[0]
        self.y = P0 * self._t1[1] + Q0 * self._t2[1] + R0 * self._t3[1]
        self.z = P0 * self._t1[2] + Q0 * self._t2[2] + R0 * self._t3[2]

        self.epsr_f: Optional[Callable] = None
        self.mur_f: Optional[Callable] = None

    def setup(self):
        """Set the material properties."""
        if self.epsr_f:
            self.epsr_f()
        if self.mur_f:
            self.mur_f()

    def plot(self):
        """Plot ep_s and mu_r."""
        raise NotImplementedError

    def set_epsr_f(self, func: Callable):
        """Set decorator to set a eps_r profile."""
        self.epsr_f = func
        return func

    def set_mur_f(self, func: Callable):
        """Set decorator to set a mu_r profile."""
        self.mur_f = func
        return func
