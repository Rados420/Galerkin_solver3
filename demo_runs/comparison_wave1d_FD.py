import numpy as np
import matplotlib.pyplot as plt
from time import time

# --- Problem setup ---
c = 1.0  # wave speed
nx = 91  # number of spatial grid points
x = np.linspace(0.0, 1.0, nx)
dx = x[1] - x[0]

CFL = 0.95  # CFL <= 1 for stability
dt = CFL * dx / c
tmax = 15.0
nt = int(np.round(tmax / dt))
dt = tmax / nt  # adjust so we end exactly at tmax

# --- Initial data ---
f_init = lambda x: np.sin(np.pi * x)  # u(x,0)
g_init = lambda x: 0.0 * x  # u_t(x,0)
u_exact = lambda x, t: np.sin(np.pi * x) * np.cos(c * np.pi * t)

# --- Initialize fields ---
u = f_init(x).copy()
v = g_init(x).copy()


def accel(u):
    """Compute acceleration (second spatial derivative)."""
    a = np.zeros_like(u)
    a[1:-1] = c**2 * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return a


# Enforce BCs initially
u[0] = 0.0
u[-1] = 0.0
v[0] = 0.0
v[-1] = 0.0

# --- Velocity Verlet time integration ---
a = accel(u)
for _ in range(nt):
    v_half = v + 0.5 * dt * a
    u[1:-1] += dt * v_half[1:-1]
    u[0] = 0.0
    u[-1] = 0.0
    a = accel(u)
    v = v_half + 0.5 * dt * a
    v[0] = 0.0
    v[-1] = 0.0

# --- Exact solution and errors ---
u_true = u_exact(x, tmax)
abs_err = np.abs(u - u_true)
L2_err = np.sqrt(np.sum((u - u_true) ** 2) * dx)
L2_true = np.sqrt(np.sum(u_true**2) * dx)
rel_L2 = L2_err / (L2_true + 1e-15)

print(f"CFL = {c*dt/dx:.3f}, nx = {nx}, dt = {dt:.5g}, tmax = {tmax}")
print(f"L2 error       = {L2_err:.6e}")
print(f"Relative L2 err= {rel_L2:.6e}")

# --- Plot results ---
plt.figure(figsize=(8, 5))
plt.plot(x, u_true, "k--", linewidth=2, label="Exact")
plt.plot(x, u, "b-", label="Numerical")
plt.title(f"1D Wave Equation at t={tmax:.2f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x, abs_err, "r", label="|u - u_exact|")
plt.title(f"Absolute Error at t={tmax:.2f}")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
