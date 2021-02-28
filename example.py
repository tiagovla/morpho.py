"""3D example."""

from pwe import BrillouinZonePath as BZPath
from pwe import Geometry, Solver
from pwe import SymmetryPoint as SPoint

# input:
Nx, Ny, Nz = 128, 128, 128
P, Q, R = 3, 3, 3

a = 1
w = 0.2 * a
eps_r = 2.34
mu_r = 1.0

G = SPoint((0, 0, 0), "Γ")
Z = SPoint((0, 0, 1 / 2), "Z")
X = SPoint((1 / 2, 0, 0), "X")
D = SPoint((1 / 2, 1 / 2, 1 / 2), "D")
T = SPoint((1 / 2, 0, 1 / 2), "T")

bz_path = BZPath([D, Z, G, Z, T, X], 50)

t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)
geo = Geometry(t1, t2, t3, Nx, Ny, Nz)


@geo.set_epsr_f
def epsr_f():
    """Define eps_r profile function."""
    mask = geo.x**2 + geo.y**2 + geo.z**2 < 0.2**2
    geo.eps_r[mask] = eps_r


@geo.set_mur_f
def mur_f():
    """Define mu_r profile function."""
    pass


solver = Solver(geometry=geo, path=bz_path, P=P, Q=Q, R=R)
solver.run()
