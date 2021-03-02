"""This module implements utilities."""
import matplotlib.pyplot as plt
import numpy as np


def convmat(A: np.ndarray, P: int = 1, Q: int = 1, R: int = 1):
    """Create a convolution matrix."""
    NH = P * Q * R
    Nx, Ny, Nz = A.shape

    p = np.arange(-(P // 2), P//2 + 1)
    q = np.arange(-(Q // 2), Q//2 + 1)
    r = np.arange(-(R // 2), R//2 + 1)

    A = np.fft.fftshift(np.fft.fftn(A) / (Nx*Ny*Nz))

    p_0 = Nx // 2
    q_0 = Ny // 2
    r_0 = Nz // 2

    C = np.zeros((NH, NH), dtype=complex)

    for r_row in range(R):
        for q_row in range(Q):
            for p_row in range(P):
                row = r_row*Q*P + q_row*P + p_row

                for r_col in range(R):
                    for q_col in range(Q):
                        for p_col in range(P):
                            col = r_col*Q*P + q_col*P + p_col

                            p_fft = p[p_row] - p[p_col]
                            q_fft = q[q_row] - q[q_col]
                            r_fft = r[r_row] - r[r_col]

                            C[row, col] = A[p_0 + p_fft, q_0 + q_fft,
                                            r_0 + r_fft]
    return C
