import numpy as np
from typing import List
from src.basis.basis_generation import Element
from scipy.integrate import quad
from functools import reduce


def assemble_matrix_integral_1d(basis1: List[Element], basis2: List[Element]):
    """
     Assemble matrix A_ij = ∫ f_i(x) g_j(x) dx
     Works with basis in format List[[Element]].
    'function_num' must be precomputed.
    """

    assert len(basis1) == len(basis2), "Bases must have same structure"

    n = len(basis1)
    A = np.zeros((n, n))

    for i, ei in enumerate(basis1):
        fi = ei["function_num"]
        (ai, bi) = ei["support"][0]
        for j, ej in enumerate(basis2):
            fj = ej["function_num"]
            (aj, bj) = ej["support"][0]

            # overlap interval
            a, b = max(ai, aj), min(bi, bj)
            if a < b:
                val, _ = quad(lambda x: fi(x) * fj(x), a, b, epsabs=1e-12, epsrel=1e-12)
                A[i, j] = val

    return A


def extend_mass(M1d, d: int):
    """Extend 1D mass matrix M1d to d dimensions via tensor (Kronecker) products."""
    M = M1d
    for _ in range(d - 1):
        M = np.kron(M, M1d)
    return M


def extend_stiffness(M1d, S1d, d: int):
    """Extend 1D stiffness and mass matrices to d dimensions (isotropic form)."""
    # Sum of Kronecker products with S1d appearing once in each dimension
    terms = []
    for i in range(d):
        factors = [M1d] * d
        factors[i] = S1d
        term = reduce(np.kron, factors)
        terms.append(term)
    return sum(terms)


if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.operators import differentiate
    from primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()

    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_0=2, J_Max=2, comp_call=True)
    b = basis_handler.flatten()
    phi_b = b[0]["function_num"]
    phi = b[1]["function_num"]
    xx = np.linspace(0, 1, 100)
    plt.plot(xx, phi_b(xx))
    plt.plot(xx, phi(xx - 1 / 4))
    plt.show()

    g00 = []
    h00 = []
    for k in range(4):
        g00.append(
            quad(lambda x: phi(x) * phi(x - k / 4), 0, 1, epsabs=1e-12, epsrel=1e-12)
        )
        h00.append(
            quad(lambda x: phi_b(x) * phi(x - k / 4), 0, 1, epsabs=1e-12, epsrel=1e-12)
        )

    print(g00)
    print(h00)

    start = time()
    M = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    basis_handler.apply(differentiate, axis=0)
    S = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    end = time()
    # S = extend_stiffness(M, S, d=2)
    # M = extend_mass(M, d=2)
    # print(f"Matrix assembled in {end-start} seconds")
    # print(f"conditional number M :  {np.linalg.cond(M)}")
    # print(f"conditional number S :  {np.linalg.cond(S)}")
    print(M.shape)
    print(M[0])
    print(M[1])
    plt.imshow(M)
    plt.show()
    plt.imshow(S)
    plt.show()
