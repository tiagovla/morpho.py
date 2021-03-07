"""3D example."""
import matplotlib.pyplot as plt
import numpy as np
from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver2D
from morpho import SymmetryPoint as SPoint

Nx, Ny = 64, 64
P, Q = 5, 5
a = 1
eps_r = 9.8
mu_r = 1.0

# Define the symmetry points
G = SPoint((0, 0), "Γ")
X = SPoint((1 / 2, 0), "X")
M = SPoint((1 / 2, 1 / 2), "M")

t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)

# Construct the bloch wave path
bz_path = BZPath([G, X, M, G], t1, t2, n_points=100)

# Construct the geometry
geo = Geometry(t1, t2, None, Nx, Ny, 1)


# Define the permitivity profile
@geo.set_epsr_f
def epsr_f():
    """Define eps_r profile function."""
    mask = geo.x**2 + geo.y**2 <= 0.2**2
    geo.eps_r[mask] = eps_r


# Define the permeability profile
@geo.set_mur_f
def mur_f():
    """Define mu_r profile function."""


# Solve
solver_tm = Solver2D(geometry=geo, path=bz_path, P=P, Q=Q, pol="TM")
solver_tm.run()

solver_te = Solver2D(geometry=geo, path=bz_path, P=P, Q=Q, pol="TE")
solver_te.run()

# Plot results:
beta_len = bz_path.beta_vec_len
wn_tm = np.vstack(solver_tm.wn)
wn_te = np.vstack(solver_te.wn)
fig, ax = plt.subplots()
ax.set_xticklabels(bz_path.symmetry_names)
ax.set_xticks(bz_path.symmetry_locations)
ax.set_xlim(0, bz_path.symmetry_locations[-1])
ax.set_ylim(0, 1)
ax.set_xlabel(r"Bloch Wave Vector $\beta$")
ax.set_ylabel(r"Frequency $\frac{\omega a}{2\pi c}$")
ax.plot(beta_len, wn_tm * a / (2 * np.pi), "b-", label="TM")
ax.plot(beta_len, wn_te * a / (2 * np.pi), "r-", label="TE")
handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc="best")
ax.grid(True)
plt.tight_layout()
plt.show()
