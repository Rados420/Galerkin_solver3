import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------- RHS projection in 2D ----------
def project_rhs_2d_kron(basis1d, f, nx=60):
    """Project general f(x,y) onto tensor-product basis built from basis1d."""
    N1 = len(basis1d)
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, nx)
    wx = 1.0 / (nx - 1)
    wy = 1.0 / (nx - 1)

    # Basis evaluation tables
    Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])  # (nx, N1)
    Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])  # (ny, N1)

    # f(x,y) samples
    F = np.array([[f(x, y) for y in ys] for x in xs])

    # Approximate double integral ∬ f φ_i φ_j
    B = Phix.T @ F @ Phiy * wx * wy  # (N1, N1)
    return B.ravel()


# ---------- Simple diagonal preconditioned solver ----------
def poisson_solver(S, b):
    """Solve (D^-1/2 S D^-1/2)ũ = D^-1/2 b"""
    D = np.diag(np.diag(S))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_tilde = D_inv_sqrt @ S @ D_inv_sqrt
    f_tilde = D_inv_sqrt @ b
    u_tilde = np.linalg.solve(A_tilde, f_tilde)
    return D_inv_sqrt @ u_tilde


# ---------- Evaluate 2D solution ----------
def evaluate_solution_2d_kron(basis1d, coeffs_kron, xs, ys):
    """Evaluate u(x,y) = Σ_ij C[i,j] φ_i(x)φ_j(y)."""
    N1 = len(basis1d)
    C = coeffs_kron.reshape(N1, N1)
    Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])
    Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])
    return Phix @ C @ Phiy.T


# ---------- Demo run ----------
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from demo_poisson import project_rhs
    from src.matrix_generation import (
        assemble_matrix_integral_1d,
        extend_mass,
        extend_stiffness,
    )
    from src.primitives import Primitives_MinimalSupport
    from src.operators import differentiate

    primitives = Primitives_MinimalSupport()
    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_Max=6, J_0=2, comp_call=True)

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
    b2 = project_rhs_2d_kron(basis_handler.flatten(), f)
    print("Solving system ...")
    coeffs = poisson_solver(S2, b2)

    # Evaluate numerical & exact solutions
    xs = np.linspace(0, 1, 60)
    ys = np.linspace(0, 1, 60)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    U_num = evaluate_solution_2d_kron(basis_handler.flatten(), coeffs, xs, ys)
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
