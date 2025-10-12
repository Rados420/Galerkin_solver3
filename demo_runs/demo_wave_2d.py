import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------- Projections & eval (Kronecker world) ----------
def project_rhs_2d_kron(basis1d, f, nx=60):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, nx)
    wx = 1.0 / (nx - 1)
    wy = 1.0 / (nx - 1)
    Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])  # (nx,N1)
    Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])  # (ny,N1)
    F = np.array([[f(x, y) for y in ys] for x in xs])
    B = Phix.T @ F @ Phiy * wx * wy
    return B.ravel()


def evaluate_solution_2d_kron(basis1d, coeffs_kron, xs, ys):
    N1 = len(basis1d)
    C = coeffs_kron.reshape(N1, N1)
    Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])
    Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])
    return Phix @ C @ Phiy.T


# ---------- Diagonal scaling solve (as in the paper) ----------
def solve_precond(A, rhs):
    D = np.diag(np.diag(A))
    D_isqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_tilde = D_isqrt @ A @ D_isqrt
    rhs_tilde = D_isqrt @ rhs
    y = np.linalg.solve(A_tilde, rhs_tilde)
    return D_isqrt @ y


# ---------- Newmark (β=1/4, γ=1/2) with consistent M ----------
def wave_newmark(M, S, b_of_t, u0c, v0c, dt, T):
    β, γ = 1 / 4, 1 / 2
    steps = int(round(T / dt))
    inv_coeff = 1.0 / (β * dt * dt)

    # Constant “effective stiffness” K = S + (1/(β dt^2)) M
    K = S + inv_coeff * M

    # initial acceleration: a0 = M^{-1}( b(0) - S u0 )
    a = solve_precond(M, b_of_t(0.0) - S @ u0c)
    u = u0c.copy()
    v = v0c.copy()

    traj = [u.copy()]
    for k in range(1, steps + 1):
        t = k * dt

        # Newmark predictor
        u_pred = u + dt * v + (0.5 - β) * dt * dt * a

        # RHS for u_{n+1}: (b_{n+1} + (1/(β dt^2)) M u_pred)
        rhs = b_of_t(t) + inv_coeff * (M @ u_pred)

        # Solve for u_{n+1} with diagonal scaling (paper)
        u_next = solve_precond(K, rhs)

        # Update a_{n+1}, v_{n+1}
        a_next = inv_coeff * (u_next - u_pred)
        v_next = v + dt * ((1 - γ) * a + γ * a_next)

        # roll
        u, v, a = u_next, v_next, a_next
        traj.append(u.copy())

    return np.array(traj)


# ---------- Demo ----------
if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from src.matrix_generation import (
        assemble_matrix_integral_1d,
        extend_mass,
        extend_stiffness,
    )
    from src.primitives import Primitives_MinimalSupport
    from src.operators import differentiate
    from src.plotting import animate_surface_2D

    # 1) Build 1D basis and matrices
    primitives = Primitives_MinimalSupport()
    bh = BasisHandler(primitives=primitives, dimension=1)
    bh.build_basis(J_Max=4, J_0=2, comp_call=True)

    M1 = assemble_matrix_integral_1d(bh.flatten(), bh.flatten())
    bh_d = copy.deepcopy(bh)
    bh_d.apply(differentiate, axis=0)
    S1 = assemble_matrix_integral_1d(bh_d.flatten(), bh_d.flatten())

    # 2) 2D via Kronecker
    M2 = extend_mass(M1, 2)
    S2 = extend_stiffness(M1, S1, 2)
    print(f"M2/S2 shape: {M2.shape}")

    # 3) Initial conditions   - standing wave
    # u0_fun = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    # v0_fun = lambda x, y: 0.0
    # f_fun = lambda x, y, t: 0.0

    # 3) Initial conditions & forcing — Gaussian impulse
    x0, y0 = 0.5, 0.5  # center of the impulse
    sigma = 0.05  # standard deviation (width of the bump)

    u0_fun = lambda x, y: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    v0_fun = lambda x, y: 0.0
    f_fun = lambda x, y, t: 0.0

    # L2 projections (then mass-invert!)
    b_u0 = project_rhs_2d_kron(bh.flatten(), u0_fun)
    b_v0 = project_rhs_2d_kron(bh.flatten(), v0_fun)

    u0c = solve_precond(M2, b_u0)  # crucial: consistent mass inverse
    v0c = solve_precond(M2, b_v0)

    # forcing vector in time (here zero)
    b_of_t = lambda t: np.zeros(M2.shape[0])

    # 4) Time integrate (unconditionally stable Newmark)
    dt, T = 5e-3, 1
    print("Integrating ...")
    U_hist = wave_newmark(M2, S2, b_of_t, u0c, v0c, dt, T)

    # 5) Evaluate & compare to exact u(x,y,t)=sin(pi x) sin(pi y) cos(sqrt(2) pi t)
    # xs = np.linspace(0, 1, 80)
    # ys = np.linspace(0, 1, 80)
    # X, Y = np.meshgrid(xs, ys, indexing="ij")
    # U_num = evaluate_solution_2d_kron(bh.flatten(), U_hist[-1], xs, ys)
    # U_ex = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.sqrt(2) * np.pi * T)
    #
    # abs_err = np.abs(U_num - U_ex)
    # l2_err = np.sqrt(np.mean(abs_err**2))
    # max_err = abs_err.max()
    # print(f"L2 error  = {l2_err:.3e}")
    # print(f"Max error = {max_err:.3e}")

    # 6) Surface plot

    xs = np.linspace(0, 1, 90)
    ys = np.linspace(0, 1, 90)
    animate_surface_2D(bh.flatten(), U_hist, xs, ys, speed=5.0, frame_step=1)
