"""This module implements classes related to the brillouinzone."""

from typing import List, Tuple

import numpy as np


class SymmetryPoint:
    """Model of a symmetry point at the brillouinzone."""

    def __init__(self, point: Tuple[float, float, float], name: str):
        """Initialize a SymmetryPoint."""
        self.point = np.array(point)
        self.name = name


class BrillouinZonePath:
    """Model of a path at the brillouinzone."""

    def __init__(self, path: List[SymmetryPoint], n_points: int = 50):
        """Initialize a BrillouinZonePath."""
        self.path = path
        self.n_points = n_points
