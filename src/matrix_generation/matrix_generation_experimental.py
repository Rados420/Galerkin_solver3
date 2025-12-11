import numpy as np
from numpy.polynomial.legendre import leggauss


def compute_c_ext(X, Y, s, q_ext=12):
    """
    Compute c_ext(x) = ∫_{R^2  Ω} 1|x-y|^{2+2s} dy
    at quadrature points (X[k],Y[k]) using another Gauss grid
    on a moderately sized box containing Ω.
    """
    # outer box for exterior integration
    L = 4.0  # size; adjust if needed
    g, w = leggauss(q_ext)

    # 1D map [-1,1] → [-L,L]
    t = 0.5 * (g + 1.0)
    y1 = -L + 2 * L * t
    wy = 2 * L * w * 0.5

    Y1, Y2 = np.meshgrid(y1, y1, indexing="ij")
    Wy = wy[:, None] * wy[None, :]

    c = np.zeros_like(X)
    for k in range(len(X)):
        x0 = X[k]
        y0 = Y[k]

        dx = x0 - Y1
        dy = y0 - Y2
        R2 = dx * dx + dy * dy

        # mask away Ω
        mask_out = (Y1 < 0) | (Y1 > 1) | (Y2 < 0) | (Y2 > 1)
        R2 = R2[mask_out]
        Wy2 = Wy[mask_out]

        c[k] = np.sum(Wy2 / (R2 ** ((2 + 2 * s) / 2.0)))

    return c


def build_fractional_stiffness_matrix(elems, s: float, q: int = 256, q_ext: int = 12):
    """
    Correct integral fractional Laplacian stiffness matrix on Ω = (0,1)^2
    with zero exterior conditions.
    """
    nb = len(elems)

    # quadrature on Ω
    g, w = leggauss(q)
    x1d = 0.5 * (g + 1.0)
    w1d = 0.5 * w

    X2d, Y2d = np.meshgrid(x1d, x1d, indexing="ij")
    Wx2d, Wy2d = np.meshgrid(w1d, w1d, indexing="ij")

    X = X2d.ravel()
    Y = Y2d.ravel()
    W = (Wx2d * Wy2d).ravel()
    Np = X.size

    # compute c_ext at quadrature points
    Cext = compute_c_ext(X, Y, s, q_ext=q_ext)

    # pairwise kernel inside Ω×Ω
    dx = X[:, None] - X[None, :]
    dy = Y[:, None] - Y[None, :]
    R2 = dx * dx + dy * dy

    Wmat = W[:, None] * W[None, :]
    K = np.zeros_like(R2)
    mask = R2 > 0
    K[mask] = Wmat[mask] / (R2[mask] ** ((2 + 2 * s) / 2.0))

    # evaluate basis functions
    Phi = np.empty((nb, Np))
    for i, elem in enumerate(elems):
        Phi[i, :] = elem["function_num"](X, Y)

    A = np.zeros((nb, nb))

    # main bilinear form
    for i in range(nb):
        dphi_i = Phi[i][:, None] - Phi[i][None, :]
        for j in range(i, nb):
            dphi_j = Phi[j][:, None] - Phi[j][None, :]
            val = np.sum(dphi_i * dphi_j * K)
            # plus diagonal-like exterior term
            val += np.sum(Phi[i] * Phi[j] * Cext * W)
            A[i, j] = val
            A[j, i] = val

    return A
