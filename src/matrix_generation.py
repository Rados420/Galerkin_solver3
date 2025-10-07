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

    n = len(elems1)
    A = np.zeros((n, n))

    for i, ei in enumerate(elems1):
        fi, (ai, bi) = ei["function_num"], ei["support"]
        for j, ej in enumerate(elems2):
            fj, (aj, bj) = ej["function_num"], ej["support"]

            # overlap
            a, b = max(ai, aj), min(bi, bj)
            if a < b:
                val, _ = quad(lambda x: fi(x) * fj(x), a, b, epsabs=1e-12, epsrel=1e-12)
                A[i, j] = val
    return A


if __name__ == "__main__":
    from src.basis.basis import BaseBasis
    from primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()

    basis = BaseBasis(primitives=primitives)
    dbasis = BaseBasis(primitives=primitives)
    basis.build_basis(J_Max=5, J_0=2, dimension=1)
    dbasis.build_basis(J_Max=5, J_0=2, dimension=1)
    basis.compute_callables()

    dbasis.differentiate_basis()
    dbasis.compute_callables()

    start = time()
    M = assemble_matrix(basis.basis, basis.basis)
    S = assemble_matrix(dbasis.basis, dbasis.basis)
    end = time()
    print(f"Matrix assembled in {end-start} seconds")
    print(f"conditional number M :  {np.linalg.cond(M)}")
    print(f"conditional number S :  {np.linalg.cond(S)}")
    plt.imshow(M)
    plt.imshow(S)
    plt.show()
