import numpy as np
from scipy.integrate import quad


def assemble_matrix_integral_1d(basis1, basis2):
    """
    Assemble matrix A_ij = ∫ f_i(x) g_j(x) dx
    Works with basis in format List[Dict[str, Element]].
    Each Element must contain 'function_num' and 'support'.
    """

    # flatten basis dicts into lists of Elements
    elems1 = [elem for group in basis1 for elem in group.values()]
    elems2 = [elem for group in basis2 for elem in group.values()]

    assert len(elems1) == len(elems2), "Bases must have same structure"

    n = len(elems1)
    A = np.zeros((n, n))

    for i, ei in enumerate(elems1):
        fi = ei["function_num"]
        (ai, bi) = ei["support"][0]
        for j, ej in enumerate(elems2):
            fj = ej["function_num"]
            (aj, bj) = ej["support"][0]

            # overlap interval
            a, b = max(ai, aj), min(bi, bj)
            if a < b:
                val, _ = quad(lambda x: fi(x) * fj(x), a, b, epsabs=1e-12, epsrel=1e-12)
                A[i, j] = val

    return A


if __name__ == "__main__":
    from src.basis.basis import BasisHandler,differentiate
    from primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()

    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_0=2,J_Max=5,comp_call=True)


    start = time()
    M = assemble_matrix_integral_1d(basis_handler.basis, basis_handler.basis)
    basis_handler.apply(differentiate,axis=0)
    S = assemble_matrix_integral_1d(basis_handler.basis, basis_handler.basis)
    end = time()
    print(f"Matrix assembled in {end-start} seconds")
    print(f"conditional number M :  {np.linalg.cond(M)}")
    print(f"conditional number S :  {np.linalg.cond(S)}")
    plt.imshow(M)
    plt.imshow(S)
    plt.show()
