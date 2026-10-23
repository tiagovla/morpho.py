"""This module implements utilities."""

import numpy as np


def harmonic_indices(count: int) -> np.ndarray:
    """Return ``count`` integer harmonics centered as closely as possible at 0.

    For an even count, the integer range is necessarily asymmetric; this
    convention keeps zero and the negative Nyquist-side harmonic.
    """
    if count < 1:
        raise ValueError("The number of harmonics must be positive.")
    return np.arange(-(count // 2), (count + 1) // 2)


def convmat(A: np.ndarray, P: int = 1, Q: int = 1, R: int = 1):
    """Return a permittivity/permeability convolution matrix.

    Parameters
    ----------
    A : np.ndarray
        A 3D/2D/1D permittivity or permeability matrix.
    P : int
        Number of terms in the direction of the reciprocal vector T1.
    Q : int
        Number of terms in the direction of the reciprocal vector T2.
    R : int
        Number of terms in the direction of the reciprocal vector T3.
    """
    NH = P * Q * R
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None, None]
    elif A.ndim == 2:
        A = A[:, :, None]
    elif A.ndim != 3:
        raise ValueError("A must have one, two, or three dimensions.")
    N1, N2, N3 = A.shape

    p = harmonic_indices(P)
    q = harmonic_indices(Q)
    r = harmonic_indices(R)

    A = np.fft.fftshift(np.fft.fftn(A) / A.size)

    p_0, q_0, r_0 = N1 // 2, N2 // 2, N3 // 2

    p_, q_, r_ = np.meshgrid(p, q, r, indexing="ij")
    rr_, cc_ = np.mgrid[0:NH, 0:NH]
    rr_i = np.unravel_index(rr_, p_.shape, order="F")
    cc_i = np.unravel_index(cc_, p_.shape, order="F")

    p_fft_ = p_[rr_i] - p_[cc_i] + p_0
    q_fft_ = q_[rr_i] - q_[cc_i] + q_0
    r_fft_ = r_[rr_i] - r_[cc_i] + r_0

    return A[p_fft_, q_fft_, r_fft_]
