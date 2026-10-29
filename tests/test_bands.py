"""Gap edges, shared polarizations, normalization, and invalid spectra."""

import numpy as np
import pytest

from morpho import BandGap, BrillouinZonePath, Solution, SymmetryPoint, find_band_gaps


def make_solution(bands, path=None):
    if path is None:
        path = BrillouinZonePath(
            (1.0,), [SymmetryPoint((0.0,), "Γ"), SymmetryPoint((0.5,), "X")], n_points=2
        )
    return Solution(path, np.asarray(bands) * (2 * np.pi), [])


def test_gap_edges_ratio_and_reference_length():
    result = make_solution([[1, 4], [2, 3]])
    gaps = find_band_gaps(result)
    assert len(gaps) == 1
    assert isinstance(gaps[0], BandGap)
    assert gaps[0].lower == pytest.approx(2.0)
    assert gaps[0].upper == pytest.approx(3.0)
    assert gaps[0].gap_to_midgap_ratio == pytest.approx(0.4)
    scaled = find_band_gaps(result, lattice_constant=2.0)[0]
    assert scaled.lower == pytest.approx(4.0)
    assert scaled.upper == pytest.approx(6.0)
    assert scaled.gap_to_midgap_ratio == pytest.approx(0.4)
    assert find_band_gaps(result, n_bands=1) == []


def test_shared_gaps_intersect_polarizations():
    first = make_solution([[1, 4], [2, 3]])
    second = make_solution([[1.5, 2.8], [2.2, 3.5]], first.path)
    gap = find_band_gaps((first, second))[0]
    assert gap.lower == pytest.approx(2.2)
    assert gap.upper == pytest.approx(2.8)
    assert gap.gap_to_midgap_ratio == pytest.approx(0.24)
    closed = make_solution([[1, 2], [3, 4]], first.path)
    assert find_band_gaps((first, closed)) == []


def test_shared_gap_can_intersect_multiple_intervals():
    first = make_solution([[1, 3, 6], [2, 4, 5]])
    second = make_solution([[2, 5], [2.5, 4.5]], first.path)
    gaps = find_band_gaps((first, second))
    np.testing.assert_allclose(
        [(gap.lower, gap.upper) for gap in gaps], [(2.5, 3), (4, 4.5)]
    )


@pytest.mark.parametrize("upper", [2.0, 2.0 - 1e-5, 2.0 + 1e-10])
def test_touching_overlapping_and_roundoff_gaps_are_closed(upper):
    assert find_band_gaps(make_solution([[1, upper], [2, 3]])) == []


def test_complex_dtype_with_real_values_is_allowed():
    assert len(find_band_gaps(make_solution([[1 + 0j, 4], [2, 3]]))) == 1


def test_lossy_spectrum_has_no_conventional_gap():
    with pytest.raises(ValueError, match="real frequencies"):
        find_band_gaps(make_solution([[1 - 0.1j, 4 - 0.2j], [2 - 0.1j, 3 - 0.2j]]))


def test_band_limit_does_not_hide_loss_in_the_input_spectrum():
    result = make_solution([[1, 3, 5 - 0.1j], [2, 4, 6 - 0.2j]])
    with pytest.raises(ValueError, match="real frequencies"):
        find_band_gaps(result, n_bands=2)


@pytest.mark.parametrize(
    "bands",
    [
        np.ones((3, 2)),
        np.empty((2, 0)),
        np.ones(2),
        [[1, np.nan], [2, 3]],
        [[1, 4], [-1, 3]],
        [[4, 1], [2, 3]],
    ],
)
def test_reject_invalid_band_arrays(bands):
    with pytest.raises(ValueError, match="Solution bands"):
        find_band_gaps(make_solution(bands))


def test_reject_different_sampling_paths():
    first = make_solution([[1, 4], [2, 3]])
    second = make_solution([[1, 4], [2, 3]])
    second.path.path_points[-1].point[:] = 0.4
    with pytest.raises(ValueError, match="same path"):
        find_band_gaps((first, second))


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True, "1"])
def test_reject_invalid_reference_length(value):
    with pytest.raises(ValueError, match="lattice_constant"):
        find_band_gaps(make_solution([[1, 4], [2, 3]]), lattice_constant=value)


def test_reject_empty_input_and_non_solution():
    with pytest.raises(ValueError, match="At least one"):
        find_band_gaps([])
    with pytest.raises(TypeError, match="Solution"):
        find_band_gaps(np.ones((2, 2)))


@pytest.mark.parametrize("n_bands", [0, -1, 0.5, True])
def test_reject_invalid_analysis_band_count(n_bands):
    with pytest.raises(ValueError, match="n_bands"):
        find_band_gaps(make_solution([[1, 4], [2, 3]]), n_bands=n_bands)
