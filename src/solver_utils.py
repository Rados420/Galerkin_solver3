import numpy as np


def solve_precond(S, b):
    """Solve (D^-1/2 S D^-1/2)ũ = D^-1/2 b"""
    D = np.diag(np.diag(S))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_tilde = D_inv_sqrt @ S @ D_inv_sqrt
    f_tilde = D_inv_sqrt @ b
    u_tilde = np.linalg.solve(A_tilde, f_tilde)
    return D_inv_sqrt @ u_tilde


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
