"""3D example."""
from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint

Nx, Ny, Nz = 128, 128, 128
P, Q, R = 3, 3, 3

a = 1
w = 0.2 * a
eps_r = 2.34
mu_r = 1.0

# Define the symmetry points
G = SPoint((0, 0, 0), "Γ")
Z = SPoint((0, 0, 1 / 2), "Z")
X = SPoint((1 / 2, 0, 0), "X")
D = SPoint((1 / 2, 1 / 2, 1 / 2), "D")
T = SPoint((1 / 2, 0, 1 / 2), "T")

t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)

# Construct the bloch wave path
bz_path = BZPath([D, Z, G, Z, T, X], t1, t2, t3, 200)

# Construct the geometry
geo = Geometry(t1, t2, t3, Nx, Ny, Nz)


# Define the permitivity profile
@geo.set_epsr_f
def epsr_f():
    """Define eps_r profile function."""
    mask1 = (abs(geo.x) >= a/2 - w/2) & (abs(geo.y) >= a/2 - w/2)
    mask2 = (abs(geo.x) >= a/2 - w/2) & (abs(geo.z) >= a/2 - w/2)
    mask3 = (abs(geo.y) >= a/2 - w/2) & (abs(geo.z) >= a/2 - w/2)
    geo.eps_r[mask1 | mask2 | mask3] = eps_r


# Define the permeability profile
@geo.set_mur_f
def mur_f():
    """Define mu_r profile function."""


# Solve
solver = Solver(geometry=geo, path=bz_path, P=P, Q=Q, R=R)
solver.run()
solver.plot_bands()
