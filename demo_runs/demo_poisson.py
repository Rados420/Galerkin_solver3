import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp

x = sp.Symbol("x")


def project_rhs(basis_num, f):
    elems = [e for g in basis_num for e in g]
    b = np.zeros(len(elems))
    for i, ei in enumerate(elems):
        fi, (a, b_supp) = ei["function_num"], ei["support"]
        if a < b_supp:
            val, _ = quad(
                lambda xx: f(xx) * fi(xx), a, b_supp, epsabs=1e-12, epsrel=1e-12
            )
            b[i] = val
    return b


def poisson_solver_1d(S, basis_num, f):
    b = project_rhs(basis_num, f)

    D = np.diag(np.diag(S))
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)

    A_tilde = D_sqrt_inv @ S @ D_sqrt_inv
    print(f"A_tilde cond number: {np.linalg.cond(A_tilde)}")
    f_tilde = D_sqrt_inv @ b

    u_tilde = np.linalg.solve(A_tilde, f_tilde)
    u = D_sqrt_inv @ u_tilde
    return u


def evaluate_solution(basis_num, coeffs, xs):
    elems = [e for g in basis_num for e in g]
    return np.array(
        [sum(c * e["function_num"](xx) for c, e in zip(coeffs, elems)) for xx in xs]
    )


# --- Example usage ---
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.matrix_generation import assemble_matrix_intgral
    from src.primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()

    basis = BasisHandler(primitives=primitives)
    dbasis = BasisHandler(primitives=primitives)
    basis.build_basis(J_Max=4, J_0=2, dimension=1)
    dbasis.build_basis(J_Max=4, J_0=2, dimension=1)
    basis._compute_callables()

    dbasis.differentiate_basis()
    dbasis._compute_callables()

    start = time()
    S = assemble_matrix_intgral(dbasis.basis, dbasis.basis)
    end = time()

    print(f"Matrix assembled in {end - start} seconds")

    # RHS: f(x) = 1
    f = lambda x: 1.0

    coeffs = poisson_solver_1d(S, basis.basis, f)

    # Evaluate numerical solution
    xs = np.linspace(0, 1, 400)
    u_num = evaluate_solution(basis.basis, coeffs, xs)

    # Exact solution
    u_exact = 0.5 * xs * (1 - xs)

    # Plot
    plt.plot(xs, u_exact, "k--", label="Exact $u(x)=x(1-x)/2$")
    plt.plot(xs, u_num, "r-", label="Galerkin approx")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
