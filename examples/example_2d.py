"""2D example."""
import matplotlib.pyplot as plt
import numpy as np
from morpho import BrillouinZonePath as BZPath
from morpho import Geometry, Solver2D
from morpho import SymmetryPoint as SPoint

# Input parameters
N1, N2 = 64, 64  # discretization
P, Q = 7, 7  # number of fourier terms
a = 1  # normalized lattice length
r = 0.2 * a  # cylinder radius
eps_r = 9.8  # permittivity
mu_r = 1.0  # permeability

# Define the symmetry points
G = SPoint((0, 0), "Γ")
X = SPoint((1 / 2, 0), "X")
M = SPoint((1 / 2, 1 / 2), "M")

# Lattice vectors
t1, t2 = (a, 0), (0, a)

# Construct the bloch wave path
bz_path = BZPath([G, X, M, G], t1, t2, n_points=100)

# Construct the geometry
geo = Geometry(t1=t1, t2=t2, N1=N1, N2=N2)


# Define the permitivity profile
@geo.set_eps_rf
def epsr_f():
    """Define eps_r profile function."""
    mask = geo.x**2 + geo.y**2 <= 0.2**2
    geo.eps_r[mask] = eps_r


# Define the permeability profile
@geo.set_mu_rf
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

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xticklabels(bz_path.symmetry_names)
ax.set_xticks(bz_path.symmetry_locations)
ax.set_xlim(0, bz_path.symmetry_locations[-1])
ax.set_ylim(0, 0.8)
ax.set_xlabel(r"Bloch Wave Vector $\beta$")
ax.set_ylabel(r"Frequency ${\omega a}/{2\pi c}$")
ax.plot(beta_len, wn_tm * a / (2 * np.pi), "b-", label="TM")
ax.plot(beta_len, wn_te * a / (2 * np.pi), "r-", label="TE")
handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc="best")
ax.grid(True)
plt.tight_layout()
plt.show()
