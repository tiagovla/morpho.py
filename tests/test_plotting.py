"""Check plotted frequency components and lossless-only gap shading."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from morpho import BrillouinZonePath, Solution, SymmetryPoint, plot_bands


@pytest.mark.parametrize("component", ["real", "imag"])
def test_complex_bands_plot_requested_component_without_gaps(component):
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    frequencies = np.array([[1 - 0.1j, 3 - 0.2j], [2 - 0.15j, 4 - 0.25j]])
    result = Solution(path, frequencies, [])
    axis = plot_bands(result, component=component)
    expected = getattr(frequencies, component) / (2 * np.pi)
    np.testing.assert_allclose(axis.lines[0].get_ydata(), expected[:, 0])
    np.testing.assert_allclose(axis.lines[1].get_ydata(), expected[:, 1])
    assert not axis.patches
    assert ("Real" if component == "real" else "Imaginary") in axis.get_ylabel()
    plt.close(axis.figure)
