import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ---------- RHS projection ----------
def project_rhs(basis_num, f):
    """Project scalar function f(x) onto 1D basis (list of Element dicts)."""
    b = np.zeros(len(basis_num))
    for i, ei in enumerate(basis_num):
        fi, (a, b_supp) = ei["function_num"], ei["support"][0]
        if a < b_supp:
            val, _ = quad(lambda x: f(x) * fi(x), a, b_supp, epsabs=1e-12, epsrel=1e-12)
            b[i] = val
    return b


# ---------- Wave equation solver (1D, preconditioned, consistent) ----------
def wave_solver_1d(M, S, basis_num, f_init, g_init, T=1.0, dt=1e-3):
    """
    Solve u_tt = u_xx on (0,1), with homogeneous Dirichlet BCs via wavelet basis.
    Uses the diagonal scaling preconditioning (Section 5 of paper) and consistent mass.
    Time integration: velocity–Verlet in preconditioned coordinates.
    """
    # 1) Project and mass-invert initial data
    b_f = project_rhs(basis_num, f_init)
    b_g = project_rhs(basis_num, g_init)
    u0 = np.linalg.solve(M, b_f)
    v0 = np.linalg.solve(M, b_g)

    # 2) Diagonal preconditioning
    D = np.diag(np.diag(S))
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D)))
    D_sqrt = np.linalg.inv(D_sqrt_inv)
    M_t = D_sqrt_inv @ M @ D_sqrt_inv
    S_t = D_sqrt_inv @ S @ D_sqrt_inv

    # Transform initial data to tilde coordinates
    w = D_sqrt @ u0
    z = D_sqrt @ v0

    # 3) Time stepping (velocity–Verlet / symplectic)
    nsteps = int(np.round(T / dt))
    dt2 = dt * dt
    a = np.linalg.solve(M_t, -S_t @ w)

    W_hist = [w.copy()]
    start = time.time()
    for _ in range(nsteps):
        w_half = w + dt * z + 0.5 * dt2 * a
        a_next = np.linalg.solve(M_t, -S_t @ w_half)
        z_next = z + 0.5 * dt * (a + a_next)
        w, z, a = w_half, z_next, a_next
        W_hist.append(w.copy())

    # 4) Back-transform to original coefficients
    U_hist = [D_sqrt_inv @ wk for wk in W_hist]
    end = time.time()
    print(f"Integration time {end - start}")
    return np.array(U_hist)


# ---------- Evaluate Galerkin expansion ----------
def evaluate_solution(basis_num, coeffs, xs):
    return np.array(
        [sum(c * e["function_num"](x) for c, e in zip(coeffs, basis_num)) for x in xs]
    )


# ---------- Demo run ----------
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.matrix_generation import assemble_matrix_integral_1d
    from src.primitives import Primitives_MinimalSupport
    from src.operators import differentiate
    from src.plotting import animate_solution_1D
    import time

    primitives = Primitives_MinimalSupport()

    # Build basis and its derivative
    basis_handler = BasisHandler(primitives=primitives, dimension=1)
    basis_handler.build_basis(J_Max=6, J_0=2, comp_call=True)
    basis_diff = BasisHandler(primitives=primitives, dimension=1)
    basis_diff.build_basis(J_Max=6, J_0=2, comp_call=True)
    basis_diff.apply(differentiate, axis=0)

    # Assemble mass and stiffness matrices
    start = time.time()
    M = assemble_matrix_integral_1d(basis_handler.flatten(), basis_handler.flatten())
    S = assemble_matrix_integral_1d(basis_diff.flatten(), basis_diff.flatten())
    end = time.time()
    print(f"Matrix assembly took {end - start:.3f} s  |  size {M.shape}")

    # Initial conditions (1D eigenmode)
    f_init = lambda x: np.sin(np.pi * x)  # u(x,0)
    g_init = lambda x: 0.0 * x  # u_t(x,0)

    # --- Initial conditions: traveling Gaussian pulse to the right ---
    # x0 = 0.3  # initial pulse center
    # sigma = 0.05  # width of the pulse
    # c = 1.0  # wave speed
    #
    # # u(x,0): Gaussian pulse
    # f_init = lambda x: np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    #
    # # u_t(x,0): choose time derivative so it travels to the right
    # g_init = (
    #     lambda x: -(c / sigma**2) * (x - x0) * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    # )

    # Solve
    T, dt = 15, 1e-3
    start = time.time()
    U_hist = wave_solver_1d(M, S, basis_handler.flatten(), f_init, g_init, T=T, dt=dt)
    end = time.time()
    print(f"Integration time {end-start}")

    # Evaluate numerical and exact solutions
    xs = np.linspace(0, 1, 400)
    u_num = evaluate_solution(basis_handler.flatten(), U_hist[-1], xs)
    u_exact = np.sin(np.pi * xs) * np.cos(np.pi * T)

    # L2 error
    err_L2 = np.sqrt(np.trapz((u_num - u_exact) ** 2, xs))
    print(f"L2 error at t={T:.2f}: {err_L2:.3e}")

    # --- Plot solution ---
    xs = np.linspace(0, 1, 200)
    anim = animate_solution_1D(basis_handler.flatten(), U_hist, xs, interval=5)
