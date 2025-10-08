import sympy as sp

from typing import List, Dict
from src.primitives import Primitives, Primitives_MinimalSupport
from src.basis.basis_generation import Element, build_basis_1d, extend_isotropic_tensor


class BasisHandler:
    def __init__(self, primitives: Primitives, dimension: int):
        self.primitives = primitives
        self.basis: List[Dict[str, Element]] = [{}]
        self.dimension = dimension

    def build_basis(
        self,
        J_Max,
        J_0,
    ):
        basis_1d = build_basis_1d(self.primitives, J_Max, J_0)
        if self.dimension == 1:
            self.basis = basis_1d
        else:
            self.basis = extend_isotropic_tensor(basis_1d, self.dimension)

    def compute_callables(self) -> None:
        """
        Convert symbolic basis functions (sympy.Lambda) into NumPy-callable versions
        for the current dimension.
        """
        # Create the correct number of sympy symbols for dimension d
        x_syms = sp.symbols(f"x0:{self.dimension}")

        for group in self.basis:
            for elem in group.values():
                f = elem["function_sym"]
                if isinstance(f, sp.Lambda):
                    # Ensure correct arity: 1D, 2D, or 3D callable
                    expr = f(*x_syms)
                else:
                    expr = f
                elem["function_num"] = sp.lambdify(x_syms, expr, "numpy")

    def differentiate_basis(self) -> None:
        """
        Differentiate the symbolic part ('function_sym') of each Element in the basis.
        Returns a new basis with updated symbolic functions.
        """
        x = sp.Symbol("x")
        for group in self.basis:
            for elem in group:
                f = elem["function_sym"]
                # keep storing a Lambda so the rest of your code still works
                f_diff = (
                    sp.Lambda(x, sp.diff(f(x), x))
                    if isinstance(f, sp.Lambda)
                    else sp.Lambda(x, sp.diff(f, x))
                )
                elem["function_sym"] = f_diff


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    primitives = Primitives_MinimalSupport()
    mintest_basis = BasisHandler(primitives=primitives, dimension=2)
    mintest_basis.build_basis(J_Max=4, J_0=2)
    mintest_basis.compute_callables()

    # pick one element
    element = list(mintest_basis.basis[1].values())[15]
    print(element)

    f_num = element["function_num"]

    # mesh grid for plotting
    xx = np.linspace(0, 1, 200)
    yy = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(xx, yy)

    # evaluate function on the grid
    Z = f_num(X, Y)

    # surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=2, cstride=2, linewidth=0)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("f(x0, x1)")
    ax.set_title(f"2D Basis Element, scale={element['scale']}")
    plt.show()
