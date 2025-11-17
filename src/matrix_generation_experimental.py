import numpy as np
import matplotlib.pyplot as plt

"""
Wavelet global mass/stiffness assembly with cross-level blocks.

Updated to match your level sizes: n(j) = 2**(j+2)  ->  4, 8, 16, 32, ...

Fixes:
  1) Level size helper n_size(j) now returns 2**(j+2).
  2) Each refinement M_j maps level j -> j+1 with shape (n(j) × n(j+1)).
  3) Transfer to level L uses left-multiplication by M^T; use M1 only at the first step for wavelets.
  4) Reference Gram matrices S_ref, K_ref are (n(L) × n(L)).

Tests updated to use the new sizing and include extra spot checks.
If your basis includes a different boundary convention, specify it and I will adapt.
"""


def n_size(j: int) -> int:
    """Number of 1D basis functions at level j (your convention)."""
    return 2 ** (j + 2)  # 4,8,16,32,...


# def assemble_toeplitz(n, table):
#     M = np.zeros((n, n))
#     for k, v in table.items():
#         for i in range(n - k):
#             M[i, i + k] = v
#             if k > 0:
#                 M[i + k, i] = v
#     return M


def _symmetric_toeplitz(first_row):
    first_row = np.asarray(first_row)
    return np.array([np.roll(first_row, -i) for i in range(len(first_row))])


def _wrap_matrix(v, M):
    v = np.asarray(v)
    M = np.asarray(M)
    n = M.shape[0]

    out = np.empty((n + 2, n + 2), dtype=M.dtype)
    out[1:-1, 1:-1] = M
    out[0, :] = v
    out[:, 0] = v
    out[-1, :] = v[::-1]
    out[:, -1] = v[::-1]

    return out


def boundary_toeplitz(rows: list[list]):
    """TODO write desc here"""
    M = _symmetric_toeplitz(rows[0])
    # print(rows[0])
    # print(M.shape)
    if len(rows) == 1:
        return M
    for row in rows[1:]:
        M = _wrap_matrix(row, M)
    return M


def _extend_zero_right(arr, n):
    if len(arr) < n:
        arr = arr + [0] * (n - len(arr))
    return arr


# -----------------------------
# Refinement data (schematic masks)
# -----------------------------
# Interior scaling and wavelet masks (coarse -> fine)
h = np.array([1 / 4, 3 / 4, 3 / 4, 1 / 4])
g = np.array([-0.5, 0.5])
# Left boundary rows; right boundary is mirrored
hL = np.array([[0.5, 1.125, 0.375, 0.0]])
gL = np.array([[-0.5, 0.5, 0.0, 0.0]])


def build_refinement_matrices(J: int):
    """Build Mj0, Mj1 for levels j = 0..J-1.
    Each Mj* maps coefficients from level j to level j+1.
    Shapes: Mj0[j].shape == (n_size(j), n_size(j+1)).
    """
    Mj0, Mj1 = [], []
    for j in range(J):
        ncoarse, nfine = n_size(j), n_size(j + 1)
        M0 = np.zeros((ncoarse, nfine))
        M1 = np.zeros((ncoarse, nfine))
        # interior rows (1..ncoarse-2)
        for k in range(1, ncoarse - 1):
            # scaling mask to fine indices centered at 2*k
            for m, val in enumerate(h):
                idx = 2 * k + m - 2
                if 0 <= idx < nfine:
                    M0[k, idx] += val
            # wavelet mask to fine indices centered at 2*k
            for m, val in enumerate(g):
                idx = 2 * k + m - 2
                if 0 <= idx < nfine:
                    M1[k, idx] += val
        # left boundary (truncate if nfine smaller than mask)
        na = min(len(hL[0]), nfine)
        M0[0, :na] = hL[0, :na]
        na = min(len(gL[0]), nfine)
        M1[0, :na] = gL[0, :na]
        # right boundary = mirror of left
        M0[-1, :] = np.flip(M0[0, :])
        M1[-1, :] = np.flip(M1[0, :])
        Mj0.append(M0)
        Mj1.append(M1)
    return Mj0, Mj1


# ---------------------------------------------
# Transfer matrices to a common scaling level L
# ---------------------------------------------


def transfer_matrix(level_from: int, level_to: int, kind: str, Mj0, Mj1):
    """Return T(level_from -> level_to) with shape (n(level_to), n(level_from)).

    Logic:
      - M_j maps level j -> j+1.
      - To express A_j in level-L scaling basis, multiply transposes left to right:
            T = M_{j,a}^T * M_{j+1,0}^T * ... * M_{L-1,0}^T,
        where a = 0 for scaling, and a = 1 only at the first step for wavelets.
    """
    if level_from == level_to:
        return np.eye(n_size(level_from))

    # Start with identity of size n(level_from)
    T = np.eye(n_size(level_from))
    for m in range(level_from, level_to):
        M = Mj1[m] if (m == level_from and kind == "psi") else Mj0[m]
        # M.T shape: (n(m+1) × n(m)); T shape: (n(m) × n(level_from))
        T = M.T @ T  # result: (n(m+1) × n(level_from))
    return T  # final shape: (n(level_to) × n(level_from))


def cross_level(kindA, kindB, j, k, base_matrix, Mj0, Mj1, L_ref):
    """Compute <A_j, B_k> or <∂A_j, ∂B_k> given base matrix at L_ref."""
    TA = transfer_matrix(j, L_ref, kindA, Mj0, Mj1)  # (n(L) × n(j))
    TB = transfer_matrix(k, L_ref, kindB, Mj0, Mj1)  # (n(L) × n(k))
    # base_matrix must be (n(L) × n(L)); result is (n(j) × n(k))
    return TA.T @ base_matrix @ TB


# ----------------------------------
# Reference Gram matrices (schematic)
# ----------------------------------


def build_reference_grams(L_ref: int):
    # placeholders for demonstration only (banded SPD structure)
    s00 = {0: 0.75, 1: 0.3125, 2: 0.0125}
    g00 = {0: 0.55, 1: 0.55, 2: 0.0614583333}
    nL = n_size(L_ref)
    # S = assemble_toeplitz(nL, s00)
    # K = assemble_toeplitz(nL, g00)
    S = 0
    K = 0
    # boundary tweaks to keep SPD; replace with exact CDV overlaps in production
    S[0, 0] *= 1.1
    S[-1, -1] *= 1.1
    K[0, 0] *= 1.2
    K[-1, -1] *= 1.2
    return S, K


# ----------------------------------
# Global assembly for [Phi_j0, Psi_j0, ..., Psi_{J-1}]
# ----------------------------------


def build_global_matrix(j0: int, J: int, S_ref, K_ref, Mj0, Mj1, L_ref: int):
    # Sanity: built levels must reach L_ref
    assert L_ref <= len(
        Mj0
    ), "L_ref exceeds built levels. Build more levels or lower L_ref."

    kinds = ["phi"] + ["psi"] * (J - j0)
    levels = [j0] + list(range(j0, J))

    # number of basis functions at level j equals n_size(j)
    sizes = [n_size(j) for j in levels]

    # sanity: S_ref/K_ref size must match n(L_ref)
    nL = n_size(L_ref)
    assert S_ref.shape == (nL, nL) and K_ref.shape == (
        nL,
        nL,
    ), "Reference Grams must be n(L) × n(L)."

    N = sum(sizes)
    S_global = np.zeros((N, N))
    K_global = np.zeros((N, N))
    offsets = np.cumsum([0] + sizes)

    for a, (kindA, jA) in enumerate(zip(kinds, levels)):
        for b, (kindB, jB) in enumerate(zip(kinds, levels)):
            S_block = cross_level(kindA, kindB, jA, jB, S_ref, Mj0, Mj1, L_ref)
            K_block = cross_level(kindA, kindB, jA, jB, K_ref, Mj0, Mj1, L_ref)
            i0, i1 = offsets[a], offsets[a + 1]
            j0_, j1_ = offsets[b], offsets[b + 1]
            # sanity on block shapes
            assert S_block.shape == (
                i1 - i0,
                j1_ - j0_,
            ), "Block shape mismatch in S_global assembly."
            assert K_block.shape == (
                i1 - i0,
                j1_ - j0_,
            ), "Block shape mismatch in K_global assembly."
            S_global[i0:i1, j0_:j1_] = S_block
            K_global[i0:i1, j0_:j1_] = K_block

    return S_global, K_global


# -----------------
# Self-tests / demos
# -----------------
if __name__ == "__main__":
    # M=boundary_toeplitz([[1,2,3],[1,1,0,0,0]])
    # print(M)

    # ===Build global matrices effective===#
    # Build refinement matrices up to level 5 (maps 0->1, 1->2, ..., 5->6 exist)
    J_build = 6  # provides M for j = 0..5
    L_ref = 5  # reference level
    j0 = 1
    J_basis = 1  # basis = [Phi_{1}, Psi_{1}, Psi_{2}, Psi_{3}]

    ss_mass = _extend_zero_right(
        [0.5500000000000002, 0.21666666666666673, 0.008333333333333366, 0.0],
        n_size(L_ref) - 2,
    )
    sb_mass = _extend_zero_right([0.75, 0.3125, 0.0125, 0.0, 0.0], n_size(L_ref))
    ss_stiff = _extend_zero_right(
        [0.5500000000000002, 0.21666666666666673, 0.008333333333333366, 0.0, 0.0],
        n_size(L_ref) - 2,
    )
    sb_stiff = _extend_zero_right(
        [0.3125, 0.0, 0.3125, 0.0125, 0.0, 0.0, 0.0], n_size(L_ref)
    )

    S_ref = boundary_toeplitz([ss_stiff, sb_stiff])
    K_ref = boundary_toeplitz([ss_mass, sb_mass])

    Mj0, Mj1 = build_refinement_matrices(J_build)
    # S_ref, K_ref = build_reference_grams(L_ref)
    #
    S_global, K_global = build_global_matrix(j0, J_basis, S_ref, K_ref, Mj0, Mj1, L_ref)
    print("S_global shape:", S_global.shape)
    print("K_global shape:", K_global.shape)

    # ===Build global matrices old===#
    from src.basis.basis import BasisHandler
    from src.operators import differentiate
    from primitives import Primitives_MinimalSupport
    from src.matrix_generation import assemble_matrix_integral_1d
    from time import time

    primitives = Primitives_MinimalSupport()

    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_0=2, J_Max=2, comp_call=True)
    b = basis_handler.flatten()
    start = time()
    M = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    basis_handler.apply(differentiate, axis=0)
    S = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    end = time()

    print(M[0])

    print(K_global[0])
