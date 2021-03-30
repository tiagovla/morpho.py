"""This module implements utilities."""
import numpy as np


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
    N1, N2, N3 = A.shape

    p = np.arange(-(P // 2), P//2 + 1)
    q = np.arange(-(Q // 2), Q//2 + 1)
    r = np.arange(-(R // 2), R//2 + 1)

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
