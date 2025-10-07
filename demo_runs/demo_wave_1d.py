import numpy as np
from scipy.integrate import quad


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


def wave_solver_1d(M, S, basis_num, f_init, g_init, T=1.0, dt=1e-3):
    """
    Solve u_tt = u_xx on (0,1), homogeneous Dirichlet via basis.
    Uses diagonal preconditioning with D = diag(S) CONSISTENTLY:
      w = D^{1/2} u,  M̃ = D^{-1/2} M D^{-1/2},  S̃ = D^{-1/2} S D^{-1/2}.
    Time stepping: velocity–Verlet in (w) coordinates.
    """
    # 1) Project and mass-invert initial data to get coefficients u(0), v(0)
    b_f = project_rhs(basis_num, f_init)
    b_g = project_rhs(basis_num, g_init)
    u0 = np.linalg.solve(M, b_f)
    v0 = np.linalg.solve(M, b_g)

    # 2) Preconditioning maps
    D = np.diag(np.diag(S))
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)
    M_t = D_sqrt_inv @ M @ D_sqrt_inv  # M̃
    S_t = D_sqrt_inv @ S @ D_sqrt_inv  # S̃

    # Transform initial conditions: w0 = D^{+1/2} u0, z0 = D^{+1/2} v0
    w = D_sqrt @ u0
    z = D_sqrt @ v0

    # 3) Velocity–Verlet in tilde coordinates: M̃ ẅ + S̃ w = 0
    nsteps = int(np.round(T / dt))
    dt2 = dt * dt
    # a^0
    a = np.linalg.solve(M_t, -S_t @ w)

    W_hist = [w.copy()]
    for _ in range(nsteps):
        w_half = w + dt * z + 0.5 * dt2 * a
        a_next = np.linalg.solve(M_t, -S_t @ w_half)
        z_next = z + 0.5 * dt * (a + a_next)
        w, z, a = w_half, z_next, a_next
        W_hist.append(w.copy())

    # 4) Back to original coefficients: u = D^{-1/2} w
    U_hist = [D_sqrt_inv @ wk for wk in W_hist]
    return np.array(U_hist)


def evaluate_solution(basis_num, coeffs, xs):
    elems = [e for g in basis_num for e in g]
    return np.array(
        [sum(c * e["function_num"](xx) for c, e in zip(coeffs, elems)) for xx in xs]
    )


# --- Example usage / demo ---
if __name__ == "__main__":
    from src.basis.basis import BaseBasis
    from src.matrix_generation import assemble_matrix_intgral
    from src.primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt
    from time import time

    primitives = Primitives_MinimalSupport()

    basis = BaseBasis(primitives=primitives)
    dbasis = BaseBasis(primitives=primitives)
    basis.build_basis(J_Max=4, J_0=2, dimension=1)
    dbasis.build_basis(J_Max=4, J_0=2, dimension=1)
    basis.compute_callables()

    dbasis.differentiate_basis()
    dbasis.compute_callables()

    start = time()
    M = assemble_matrix_intgral(basis.basis, basis.basis)
    S = assemble_matrix_intgral(dbasis.basis, dbasis.basis)
    end = time()

    print(f"Matrix assembled in {end - start} seconds")

    # Initial data: exact eigenmode
    f_init = lambda x: np.sin(np.pi * x)  # u(x,0)
    g_init = lambda x: 0.0 * x  # ut(x,0)

    # Solve
    T, dt = 2.5, 1e-3
    U_hist = wave_solver_1d(M, S, basis.basis, f_init, g_init, T=T, dt=dt)

    # Evaluate numerical solution at final time
    xs = np.linspace(0, 1, 400)
    u_num = evaluate_solution(basis.basis, U_hist[-1], xs)

    # Exact solution: u(x,t) = sin(pi x) cos(pi t)
    u_exact = np.sin(np.pi * xs) * np.cos(np.pi * T)

    # L2 error norm on [0,1]
    err_L2 = np.sqrt(np.trapz((u_num - u_exact) ** 2, xs))
    print(f"L2 error at t={T}: {err_L2:.3e}")

    # Plot solution and error
    plt.figure()
    plt.plot(xs, u_exact, "k--", label="Exact $\\sin(\\pi x)\\cos(\\pi t)$")
    plt.plot(xs, u_num, "r-", label="Galerkin approx")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.title("1D Wave Equation — final time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(xs, np.abs(u_num - u_exact), label="$|u_h - u|$")
    plt.xlabel("x")
    plt.ylabel("abs error")
    plt.title(f"Pointwise absolute error in $|u_h - u|$")
    plt.grid(True)
    plt.legend()
    plt.show()
