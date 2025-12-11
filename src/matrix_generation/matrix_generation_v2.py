from typing import List, Callable
import numpy as np
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

    # ----------------------------------------------------------
    # 1. Support intersection
    # ----------------------------------------------------------
    left = []
    right = []
    for (ai, bi), (aj, bj) in zip(ei["support"], ej["support"]):
        L = max(ai, aj)
        R = min(bi, bj)
        if R <= L:
            return 0.0
        left.append(L)
        right.append(R)

    # ----------------------------------------------------------
    # 2. Grid on each dimension
    # ----------------------------------------------------------
    grids = [np.linspace(l, r, points_per_dim) for (l, r) in zip(left, right)]

    # ND mesh
    mesh = np.meshgrid(*grids, indexing="ij")

    # ----------------------------------------------------------
    # 3. Safe evaluation
    # ----------------------------------------------------------
    def f_eval(f, coords):
        # coords is a list/array-of-arrays from meshgrid
        try:
            return f(*coords)
        except TypeError:
            return f(np.array(coords))

    fi_vals = f_eval(fi, mesh)
    fj_vals = f_eval(fj, mesh)

    prod_vals = fi_vals * fj_vals

    # ----------------------------------------------------------
    # 4. Sequential ND trapezoidal reduction (always axis=0!)
    # ----------------------------------------------------------
    integral = prod_vals
    for d in range(dim):
        x = grids[d]
        integral = np.trapezoid(integral, x, axis=0)

    return float(integral)


def B_mass_1d_simple(ei: Element, ej: Element) -> float:
    """
    Very simple and robust 1D mass bilinear form:
        ∫ phi_i(x) * phi_j(x) dx
    using uniform sampling + NumPy's trapezoid integrator.
    """

    fi = ei["function_num"]
    fj = ej["function_num"]

    # compute support overlap
    ai, bi = ei["support"][0]
    aj, bj = ej["support"][0]

    left = max(ai, aj)
    right = min(bi, bj)

    if right <= left:
        return 0.0

    import numpy as np

    N = 40
    xs = np.linspace(left, right, N)

    vals = fi(xs) * fj(xs)

    return float(np.trapezoid(vals, xs))


def assemble_matrix(
    basis: List[Element],
    B: Callable[[Element, Element], float],
    restrict_to_supp_overlap: bool = True,
) -> np.ndarray:
    """
    Assemble the matrix [B(e_i, e_j)] for a flattened basis.

    Parameters
    ----------
    basis : list[Element]
        Flattened list of basis elements.
    B : callable
        Bilinear form: B(e_i, e_j) -> float (e.g. mass).
    restrict_to_supp_overlap : bool, optional
        If True, entries for basis pairs with disjoint supports
        are left as zero and B is not evaluated.
    """
    n = len(basis)
    A = np.zeros((n, n), dtype=float)

    def supports_overlap(ei: Element, ej: Element) -> bool:
        # product support; assume rectangular supports in each dimension
        for (ai, bi), (aj, bj) in zip(ei["support"], ej["support"]):
            # no overlap in this dimension ⇒ no overlap at all
            if max(ai, aj) >= min(bi, bj):
                return False
        return True

    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            if restrict_to_supp_overlap and not supports_overlap(ei, ej):
                continue
            A[i, j] = B(ei, ej)

    return A


if __name__ == "__main__":
    from src.basis.basis import BasisHandler
    from primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt

    primitives = Primitives_MinimalSupport()

    bh = BasisHandler(primitives=primitives, dimension=2)
    bh.build_basis(J_0=2, J_Max=4, comp_call=True)
    basis = bh.flatten()

    def supports_overlap(ei: Element, ej: Element) -> bool:
        # product support; assume rectangular supports in each dimension
        for (ai, bi), (aj, bj) in zip(ei["support"], ej["support"]):
            # no overlap in this dimension ⇒ no overlap at all
            if max(ai, aj) >= min(bi, bj):
                return False
        return True

    A = assemble_matrix(bh.flatten(), B_mass_nd)
    plt.imshow(A)
    plt.colorbar()
    plt.show()
