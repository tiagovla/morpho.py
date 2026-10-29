"""Check plotted frequency components and lossless-only gap shading."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from morpho import (
    BrillouinZonePath,
    Solution,
    SymmetryPoint,
    find_band_gaps,
    plot_bands,
)


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


def test_lossless_plot_shades_the_same_shared_gaps_as_analysis():
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    first = Solution(path, 2 * np.pi * np.array([[1.0, 4.0], [2.0, 3.0]]), [], "TM")
    second = Solution(path, 2 * np.pi * np.array([[1.5, 2.8], [2.2, 3.5]]), [], "TE")
    gap = find_band_gaps((first, second), lattice_constant=2)[0]
    axis = plot_bands((first, second), lattice_constant=2)
    assert len(axis.patches) == 1
    patch = axis.patches[0]
    vertices = patch.get_transform().transform(patch.get_path().vertices)
    data_vertices = axis.transData.inverted().transform(vertices)
    assert data_vertices[:, 1].min() == pytest.approx(gap.lower)
    assert data_vertices[:, 1].max() == pytest.approx(gap.upper)
    assert axis.get_legend_handles_labels()[1] == ["TM", "TE"]
    plt.close(axis.figure)


def test_imaginary_component_does_not_shade_lossless_gaps():
    path = BrillouinZonePath(
        (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
    )
    result = Solution(path, np.array([[1.0, 4.0], [2.0, 3.0]]), [])
    axis = plot_bands(result, component="imag")
    assert not axis.patches
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0.0, 0.0])
    plt.close(axis.figure)
