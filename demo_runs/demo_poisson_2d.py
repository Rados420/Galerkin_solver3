import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from mpl_toolkits.mplot3d import Axes3D
from typing import Sequence

# def _finest_level_from_basis(basis1d):
#     levels = [e.get("scale", None) for e in basis1d]  # 'scale' is in your Element
#     levels = [L for L in levels if L is not None]
#     return max(levels) if levels else 8
#
#
# def _gauss_on_dyadic(level: int, q: int):
#     """
#     Build tensorable 1D Gauss–Legendre nodes and weights on a dyadic mesh of size m=2**level.
#     Returns concatenated nodes x in (0,1) and weights w of length m*q.
#     """
#     m = 2**level
#     xi, wi = leggauss(q)
#     h = 1.0 / m
#     a = np.arange(m) * h
#     b = a + h
#     mid = (a + b)[:, None] * 0.5
#     rad = (b - a)[:, None] * 0.5
#     X = mid + rad * xi[None, :]
#     W = rad * wi[None, :]
#     return X.reshape(-1), W.reshape(-1)
#
#
# def _eval_basis_on_nodes(basis1d: Sequence[dict], nodes: np.ndarray) -> np.ndarray:
#     """
#     Φ[i,j] = φ_j(x_i) for 1D basis.
#     Uses the first interval of Element['support'] which is a tuple of tuples for general-d.
#     """
#     n_nodes = nodes.size
#     n_funcs = len(basis1d)
#     Ph = np.zeros((n_nodes, n_funcs))
#
#     for j, e in enumerate(basis1d):
#         f = e["function_num"]
#         if f is None:
#             raise ValueError("Element['function_num'] is None; provide a numeric callable.")
#         supp = e.get("support", None)
#         if not supp or len(supp) == 0:
#             # If support missing, assume whole [0,1]
#             a, b = 0.0, 1.0
#         else:
#             # 1D: take the first interval (a,b) from the tuple of tuples
#             a, b = supp[0]
#         mask = (nodes >= a) & (nodes <= b)
#         if np.any(mask):
#             Ph[mask, j] = f(nodes[mask])
#         # outside support stays zero
#     return Ph
#
#
# def project_rhs_2d_gauss(basis1d, f, level=None, qx=5, qy=5):
#     """
#     More precise yet fast projection: tensor-product Gauss–Legendre on a dyadic mesh.
#     Returns the vectorized RHS in Kronecker ordering (length N1*N1).
#     """
#     if level is None:
#         level = _finest_level_from_basis(basis1d) + 1  # a bit finer than the basis
#
#     xs, wx = _gauss_on_dyadic(level, qx)
#     ys, wy = _gauss_on_dyadic(level, qy)
#
#     Phix = _eval_basis_on_nodes(basis1d, xs)  # (nx, N1)
#     Phiy = _eval_basis_on_nodes(basis1d, ys)  # (ny, N1)
#
#     # Evaluate f on tensor grid and weight it
#     F = np.empty((xs.size, ys.size))
#     for i, x in enumerate(xs):
#         F[i, :] = f(x, ys)
#     F *= wx[:, None] * wy[None, :]
#
#     # Tensor quadrature: (N1 x nx) @ (nx x ny) @ (ny x N1) -> (N1 x N1)
#     B = Phix.T @ F @ Phiy
#     return B.ravel(order="C")


# ---------- Simple diagonal preconditioned solver ----------
def poisson_solver(S, b):
    """Solve (D^-1/2 S D^-1/2)ũ = D^-1/2 b"""
    D = np.diag(np.diag(S))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_tilde = D_inv_sqrt @ S @ D_inv_sqrt
    f_tilde = D_inv_sqrt @ b
    u_tilde = np.linalg.solve(A_tilde, f_tilde)
    return D_inv_sqrt @ u_tilde


# # ---------- Evaluate 2D solution ----------
# def evaluate_solution_2d_kron(basis1d, coeffs_kron, xs, ys):
#     """Evaluate u(x,y) = Σ_ij C[i,j] φ_i(x)φ_j(y)."""
#     N1 = len(basis1d)
#     C = coeffs_kron.reshape(N1, N1)
#     Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])
#     Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])
#     return Phix @ C @ Phiy.T


# ---------- Demo run ----------
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.matrix_generation import (
        assemble_matrix_integral_1d,
        extend_mass,
        extend_stiffness,
    )
    from src.primitives import Primitives_MinimalSupport
    from src.operators import differentiate

    primitives = Primitives_MinimalSupport()
    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_Max=5, J_0=2, comp_call=True)

    # 1D matrices
    M1 = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    basis_diff = copy.deepcopy(basis_handler)
    basis_diff.apply(differentiate, axis=0)
    S1 = assemble_matrix_integral_1d(basis_diff.flatten(), basis_diff.flatten())

    # 2D via Kronecker tensor products
    M2 = extend_mass(M1, 2)
    S2 = extend_stiffness(M1, S1, 2)
    print(f"Matrix size: {S2.shape}")

    # Define asymmetric RHS and exact solution
    f = lambda x, y: (
        -2 * x**3 + 6 * x**2 * y - 6 * x * y**2 + 2 * x + 2 * y**3 - 2 * y
    )
    u_exact = lambda x, y: x * (1 - x) * y * (1 - y) * (x - y)

    # Project RHS and solve
    print("Projecting RHS ...")
    # b2 = project_rhs_2d_gauss(basis_handler.flatten(), f)
    b2 = basis_handler.project_rhs_2d_gauss(f=f)
    print("Solving system ...")
    coeffs = poisson_solver(S2, b2)

    # Evaluate numerical & exact solutions
    xs = np.linspace(0, 1, 60)
    ys = np.linspace(0, 1, 60)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # U_num = evaluate_solution_2d_kron(basis_handler.flatten(), coeffs, xs, ys)
    U_num = basis_handler.evaluate_solution_2d_kron(coeffs, xs, ys)
    U_exact = u_exact(X, Y)

    # Error metrics
    abs_err = np.abs(U_num - U_exact)
    l2_err = np.sqrt(np.mean(abs_err**2))
    max_err = np.max(abs_err)
    print(f"L2 error  = {l2_err:.3e}")
    print(f"Max error = {max_err:.3e}")

    # 3D surface plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, U_num, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("2D Poisson solution (Galerkin, Kronecker)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    plt.tight_layout()
    plt.show()
