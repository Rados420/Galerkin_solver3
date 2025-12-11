from src.basis.basis_generation import Element


def B_mass_nd(ei: Element, ej: Element, points_per_dim: int = 20) -> float:
    """
    Universal ND mass bilinear form:
        ∫ phi_i(x) * phi_j(x) dx    over the intersection of supports.
    Works for 1D, 2D, 3D, ...
    Uses tensor-product trapezoidal integration over overlapping support.
    """

    import numpy as np

    fi = ei["function_num"]
    fj = ej["function_num"]

    dim = len(ei["support"])

    left = []
    right = []
    for (ai, bi), (aj, bj) in zip(ei["support"], ej["support"]):
        L = max(ai, aj)
        R = min(bi, bj)
        if R <= L:
            return 0.0
        left.append(L)
        right.append(R)

    grids = [np.linspace(l, r, points_per_dim) for (l, r) in zip(left, right)]

    # ND mesh
    mesh = np.meshgrid(*grids, indexing="ij")

    def f_eval(f, coords):
        # coords is a list/array-of-arrays from meshgrid
        try:
            return f(*coords)
        except TypeError:
            return f(np.array(coords))

    fi_vals = f_eval(fi, mesh)
    fj_vals = f_eval(fj, mesh)

    prod_vals = fi_vals * fj_vals
    integral = prod_vals
    for d in range(dim):
        x = grids[d]
        integral = np.trapezoid(integral, x, axis=0)

    return float(integral)
