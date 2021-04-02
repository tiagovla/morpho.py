|PyPI License| |PyPI PyVersions| |PyPI Version| |Build Status| |DeepSource| |Codecov| |Documentation Status| |DOI|

================
 PWE Framework for 3D/2D/1D Photonic Crystal Analysis
================


Quick examples:
##############

.. code-block:: python

    """2D example."""
    import matplotlib.pyplot as plt
    import numpy as np

    from morpho import BrillouinZonePath as BZPath
    from morpho import Geometry, Solver
    from morpho import SymmetryPoint as SPoint

    # Input parameters
    a = 1  # normalized lattice length
    r = 0.2 * a  # cylinder radius
    N1, N2 = 64, 64  # discretization
    P, Q = 7, 7  # number of fourier terms
    EPS_R = 9.8  # permittivity
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
        mask = x**2 + y**2 <= 0.2**2
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


Results:
**********
.. image:: docs/_static/bandgap_diagram_2d.png
  :width: 300

.. code-block:: python

    """3D example."""
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

Results:
**********
.. image:: docs/_static/bandgap_diagram_3d.png
  :width: 300

References:
###########
[1] J. D. Joannopoulos, Ed., Photonic crystals: molding the flow of light, 2nd ed. Princeton: Princeton University Press, 2008.

 

.. |PyPI License| image:: https://img.shields.io/pypi/l/morpho.py.svg
  :target: https://pypi.python.org/pypi/morpho.py

.. |PyPI PyVersions| image:: https://img.shields.io/pypi/pyversions/morpho.py.svg
  :target: https://pypi.python.org/pypi/morpho.py

.. |PyPI Version| image:: https://img.shields.io/pypi/v/morpho.py.svg
  :target: https://pypi.python.org/pypi/morpho.py

.. |Build Status| image:: https://travis-ci.com/tiagovla/morpho.py.svg?branch=master
  :target: https://travis-ci.com/tiagovla/morpho.py

.. |DeepSource| image:: https://deepsource.io/gh/tiagovla/morpho.py.svg/?label=active+issues
  :target: https://deepsource.io/gh/tiagovla/morpho.py/?ref=repository-badge

.. |Codecov| image:: https://codecov.io/gh/tiagovla/morpho.py/branch/master/graph/badge.svg?token=QR1RMTPX0H
  :target: https://codecov.io/gh/tiagovla/morpho.py

.. |Documentation Status| image:: https://readthedocs.org/projects/morpho-py/badge/?version=latest
  :target: https://morpho-py.readthedocs.io/en/latest/?badge=latest

.. |DOI| image:: https://zenodo.org/badge/341691173.svg
   :target: https://zenodo.org/badge/latestdoi/341691173
