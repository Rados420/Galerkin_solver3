import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa


def animate_solution_1D(basis, U_hist, xs, interval=30, repeat=False, filename=None):
    """
    Fast, robust 1D animation for time evolution u(x,t).
    basis  : flattened basis (list of Element dicts with "function_num")
    U_hist : array (nt, nbasis)
    xs     : spatial grid
    """
    # --- Precompute basis evaluations Φ(x_i) = φ_i(x_i)
    nb = len(basis)
    nx = len(xs)
    Phi = np.empty((nx, nb))
    for j, e in enumerate(basis):
        Phi[:, j] = [e["function_num"](x) for x in xs]

    # --- Evaluate all time frames efficiently
    U_vals = (Phi @ U_hist.T).T  # shape (nt, nx)

    # --- Setup figure
    fig, ax = plt.subplots()
    (line,) = ax.plot(xs, U_vals[0], "r-", lw=2)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(1.1 * U_vals.min(), 1.1 * U_vals.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("1D Wave Evolution")

    def update(frame):
        line.set_ydata(U_vals[frame])
        ax.set_title(f"t-step {frame}/{len(U_vals)-1}")
        return (line,)

    anim = FuncAnimation(
        fig, update, frames=len(U_vals), interval=interval, blit=False, repeat=repeat
    )

    if filename:
        print(f"Saving animation to {filename} ...")
        anim.save(filename, writer="ffmpeg", fps=max(1, 1000 // interval), dpi=150)

    plt.show()
    return anim


# ---------- 2D ANIMATION ----------


def animate_surface_2D(basis1d, U_hist, xs, ys, speed=1.0, frame_step=1):
    """
    Robust looping 3D surface animation (macOS/PyCharm friendly).
    - Switch your MPL backend to 'QtAgg' or 'TkAgg' for smooth playback.
    - 'speed' scales the pause; 'frame_step' skips frames for faster apparent motion.
    """
    # ---- precompute Φ(x), Φ(y) and all frames ----
    N1 = len(basis1d)
    Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])
    Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    U_vals = np.array([Phix @ c.reshape(N1, N1) @ Phiy.T for c in U_hist])
    umin, umax = U_vals.min(), U_vals.max()

    # ---- figure ----
    plt.ion()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y,t)")
    ax.set_zlim(1.1 * umin, 1.1 * umax)

    # draw first frame
    surf = ax.plot_surface(X, Y, U_vals[0], cmap="viridis", rstride=1, cstride=1)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # effective delay per frame (Qt/Tk honor this)
    base_pause = 0.02
    pause = max(0.0005, base_pause / max(speed, 1e-6))
    step = max(1, int(frame_step))

    try:
        while plt.fignum_exists(fig.number):
            for i in range(0, len(U_vals), step):
                if not plt.fignum_exists(fig.number):
                    break
                # replot the surface (safe + portable)
                surf.remove()
                surf = ax.plot_surface(
                    X, Y, U_vals[i], cmap="viridis", rstride=1, cstride=1
                )
                ax.set_title(
                    f"Frame {i+1}/{len(U_vals)}  |  step={step}, speed={speed:g}"
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(pause)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()
