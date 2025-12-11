from typing import List, Callable
import numpy as np
from src.basis.basis_generation import Element
from src.matrix_generation.forms import B_mass_nd


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
    from src.primitives import Primitives_MinimalSupport
    import matplotlib.pyplot as plt

    primitives = Primitives_MinimalSupport()

    bh = BasisHandler(primitives=primitives, dimension=2)
    bh.build_basis(J_0=2, J_Max=3, comp_call=True)
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
