"""2D example."""
import matplotlib.pyplot as plt
import numpy as np

from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver
from morpho import SymmetryPoint as SPoint

# Input parameters
a = 1  # normalized lattice length
r = 0.2 * a  # cylinder radius
N1, N2 = 65, 65  # discretization
P, Q = 7, 7  # number of fourier terms
EPS_R = 8.9  # permittivity
MU_R = 1.0  # permeability

# Define the symmetry points
G = SPoint((0.0, 0.0), "Γ")
X = SPoint((0.5, 0.0), "X")
M = SPoint((0.5, 0.5), "M")

# Lattice vectors
a1, a2 = (a, 0), (0, a)

# Construct the bloch wave path
bz_path = BZPath(a1, a2, [G, X, M, G], n_points=100)

# Construct the geometry
geo = Geometry(a1, a2, N1, N2)


# Define the permitivity profile
@geo.overwrite
def eps_rf(eps_r, x, y):
    """Define eps_r profile function."""
    mask = x ** 2 + y ** 2 <= 0.2 ** 2
    eps_r[mask] = EPS_R


# Solve
solver_tm = Solver(geo, bz_path, P=P, Q=Q, pol="TM")
solver_te = Solver(geo, bz_path, P=P, Q=Q, pol="TE")

solver_tm.run()
solver_te.run()

# Results:
beta_len = bz_path.betas.cumsum
wn_tm = np.vstack(solver_tm.wn)
wn_te = np.vstack(solver_te.wn)

_, ax = plt.subplots(figsize=(5, 4))

ax.set_xticklabels(bz_path.point_names)
ax.set_xticks(bz_path.point_locations)
ax.set_xlim(0, bz_path.point_locations[-1])
ax.set_ylim(0, 0.8)
ax.set_xlabel(r"Bloch Wave Vector $k$")
ax.set_ylabel(r"Frequency ${\omega a}/{2\pi c}$")
ax.plot(beta_len, wn_tm * a / (2 * np.pi), "b-", label="TM")
ax.plot(beta_len, wn_te * a / (2 * np.pi), "r-", label="TE")
ax.grid(True)
handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax.legend(handles, labels, loc="best")

plt.tight_layout()
plt.show()
