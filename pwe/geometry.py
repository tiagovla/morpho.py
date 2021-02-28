"""Implement photonic crystal geometries."""

from typing import Callable, Optional, Tuple

import numpy as np


class Geometry:
    """Geometry class."""

    def __init__(
        self,
        t1: Tuple[float, float, float],
        t2: Tuple[float, float, float],
        t3: Tuple[float, float, float],
        Nx: int = 64,
        Ny: int = 64,
        Nz: int = 64,
        eps_r: Optional[np.ndarray] = None,
        mu_r: Optional[np.ndarray] = None,
    ):
        """Initialize the geometry obj."""
        self.t1 = np.array(t1)
        self.t2 = np.array(t2)
        self.t3 = np.array(t3)

        self.T1 = 2 * np.pi * np.cross(t2, t3) / np.dot(t1, np.cross(t2, t3))
        self.T2 = 2 * np.pi * np.cross(t3, t1) / np.dot(t1, np.cross(t2, t3))
        self.T3 = 2 * np.pi * np.cross(t1, t2) / np.dot(t1, np.cross(t2, t3))

        self.eps_r = eps_r or np.ones((Nx, Ny, Nz), dtype=complex)
        self.mu_r = mu_r or np.ones((Nx, Ny, Nz), dtype=complex)

        P0, Q0, R0 = np.meshgrid(
            np.linspace(-0.5, 0.5, Nx),
            np.linspace(-0.5, 0.5, Ny),
            np.linspace(-0.5, 0.5, Nz),
        )
        self.x = P0 * self.t1[0] + Q0 * self.t2[0] + R0 * self.t3[0]
        self.y = P0 * self.t1[1] + Q0 * self.t2[1] + R0 * self.t3[1]
        self.z = P0 * self.t1[2] + Q0 * self.t2[2] + R0 * self.t3[2]

        self.epsr_f: Optional[Callable] = None
        self.mur_f: Optional[Callable] = None

    def setup(self):
        """Set the material properties."""
        if self.epsr_f:
            self.epsr_f()
        if self.mur_f:
            self.mur_f()

    def set_epsr_f(self, func: Callable):
        """Set decorator to set a eps_r profile."""
        self.epsr_f = func
        return func

    def set_mur_f(self, func: Callable):
        """Set decorator to set a mu_r profile."""
        self.mur_f = func
        return func
