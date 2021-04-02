"""2D example."""
import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint

N1, N2, N3 = 64, 64, 64
P, Q, R = 3, 3, 3

a = 1
w = 0.2 * a
EPS_R = 2.34
MU_R = 1.0

# Define the symmetry points
G = SPoint((0, 0, 0), "Γ")
Z = SPoint((0, 0, 1 / 2), "Z")
X = SPoint((1 / 2, 0, 0), "X")
D = SPoint((1 / 2, 1 / 2, 1 / 2), "D")
T = SPoint((1 / 2, 0, 1 / 2), "T")

a1, a2, a3 = (a, 0, 0), (0, a, 0), (0, 0, a)

# Construct the bloch wave path
bz_path = BZPath(a1, a2, a3, [D, Z, G, Z, T, X], 200)

# Construct the geometry
geo = Geometry(a1, a2, a3, N1, N2, N3)


# Define the permitivity profile
@geo.overwrite
def eps_rf(eps_r, x, y, z):
    """Define eps_r profile function."""
    mask1 = (abs(x) >= a/2 - w/2) & (abs(y) >= a/2 - w/2)
    mask2 = (abs(x) >= a/2 - w/2) & (abs(z) >= a/2 - w/2)
    mask3 = (abs(y) >= a/2 - w/2) & (abs(z) >= a/2 - w/2)
    eps_r[mask1 | mask2 | mask3] = EPS_R


# Solve
solver = Solver(geo, bz_path, P=P, Q=Q, R=R)
solver.run()

# Results:
beta_len = bz_path.betas.cumsum
wn = np.vstack(solver.wn)

_, ax = plt.subplots(figsize=(5, 4))

ax.set_xticklabels(bz_path.point_names)
ax.set_xticks(bz_path.point_locations)
ax.set_xlim(0, bz_path.point_locations[-1])
ax.set_ylim(0, 1.6)
ax.set_xlabel(r"Bloch Wave Vector $k$")
ax.set_ylabel(r"Frequency ${\omega a}/{2\pi c}$")
ax.plot(beta_len, wn * a / (2 * np.pi), "k-")
ax.grid(True)

plt.tight_layout()
plt.show()
