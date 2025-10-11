import numpy as np
import copy
from scipy.integrate import quad
import matplotlib.pyplot as plt

def project_rhs(basis_num, f):
    """Compute b_i = ∫ f(x) * φ_i(x) dx"""
    b = np.zeros(len(basis_num))
    for i, ei in enumerate(basis_num):
        fi, (a, b_supp) = ei["function_num"], ei["support"][0]
        if a < b_supp:
            val, _ = quad(lambda xx: f(xx) * fi(xx), a, b_supp,
                          epsabs=1e-12, epsrel=1e-12)
            b[i] = val
    return b


def poisson_solver_1d(S, b):
    """Solve Ãũ = f̃ with simple diagonal preconditioning"""
    D = np.diag(np.diag(S))
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(S)))

    A_tilde = D_sqrt_inv @ S @ D_sqrt_inv
    f_tilde = D_sqrt_inv @ b

    u_tilde = np.linalg.solve(A_tilde, f_tilde)
    u = D_sqrt_inv @ u_tilde
    return u


def evaluate_solution(basis_num, coeffs, xs):
    """Evaluate linear combination of basis functions at points xs."""
    return np.array([
        sum(c * e["function_num"](xx) for c, e in zip(coeffs, basis_num))
        for xx in xs
    ])


# ---------------- Example usage ----------------
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.matrix_generation import assemble_matrix_integral_1d
    from src.primitives import Primitives_MinimalSupport
    from src.operators import differentiate

    primitives = Primitives_MinimalSupport()

    # build the original basis (φ, ψ)
    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_Max=4, comp_call=True, J_0=2)

    # make a copy for differentiated basis (φ′, ψ′)
    basis_handler_diff = copy.deepcopy(basis_handler)
    basis_handler_diff.apply(differentiate, comp_call=True, axis=0)

    # assemble stiffness matrix using derivatives
    S = assemble_matrix_integral_1d(basis_handler_diff.flatten(),
                                    basis_handler_diff.flatten())

    # project RHS with original basis
    f = lambda x: 1.0
    b = project_rhs(basis_handler.flatten(), f)

    # solve
    coeffs = poisson_solver_1d(S, b)

    # evaluate and plot
    xs = np.linspace(0, 1, 400)
    u_num = evaluate_solution(basis_handler.flatten(), coeffs, xs)
    u_exact = 0.5 * xs * (1 - xs)

    plt.plot(xs, u_exact, "k--", label="Exact $u(x)=x(1-x)/2$")
    plt.plot(xs, u_num, "r-", label="Galerkin approx")
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.legend(); plt.grid(True); plt.show()
