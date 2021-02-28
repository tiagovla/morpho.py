"""Implement the solver."""

from .brillouinzone import BrillouinZonePath as BZPath
from .geometry import Geometry
from .utils import convmat


class Solver:
    """Implement the PWE solver."""

    def __init__(self,
                 geometry: Geometry,
                 path: BZPath,
                 P: int = 1,
                 Q: int = 1,
                 R: int = 1):
        """Initialize the PWE Solver."""
        self.geo = geometry
        self.path = path
        self.P, self.Q, self.R = P, Q, R

    def run(self):
        """Run the simulation."""
        self.geo.setup()
        self.eps_rc = convmat(self.geo.eps_r, self.P, self.Q, self.R)
        self.mu_rc = convmat(self.geo.mu_r, self.P, self.Q, self.R)
