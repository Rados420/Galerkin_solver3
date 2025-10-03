import numpy as np
from scipy.integrate import quad


def assemble_matrix(basis1, basis2):
    """
    Assemble matrix A_ij = ∫ f_i(x) g_j(x) dx
    basis1, basis2: List[List[Element]] with same structure (supports etc.)
    """
    elems1 = [e for group in basis1 for e in group]
    elems2 = [e for group in basis2 for e in group]
    assert len(elems1) == len(elems2), "Bases must have same structure"

    xx = np.linspace(0, 1, 400)

    n = len(elems1)
    A = np.zeros((n, n))

    for i, ei in enumerate(elems1):
        fi, (ai, bi) = ei["function"], ei["support"]
        for j, ej in enumerate(elems2):
            fj, (aj, bj) = ej["function"], ej["support"]

            # overlap
            a, b = max(ai, aj), min(bi, bj)
            if a < b:
                val, _ = quad(lambda x: fi(x) * fj(x), a, b, epsabs=1e-12, epsrel=1e-12)
                A[i, j] = val
    return A


if __name__ == "__main__":
    from src.basis import build_basis_1d
    from primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()
    basis = build_basis_1d(primitives=primitives, J_max=5)
    start = time()
    M = assemble_matrix(basis, basis)
    end = time()
    print(f"Matrix assembled in {end-start} seconds")
    plt.imshow(M)
    plt.show()
