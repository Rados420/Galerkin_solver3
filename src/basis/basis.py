import sympy as sp
from typing import List, Dict, Callable
from src.primitives import Primitives, Primitives_MinimalSupport
from src.basis.basis_generation import Element, build_basis_1d, extend_isotropic_tensor


class BasisHandler:
    def __init__(self, primitives: Primitives, dimension: int):
        self.primitives = primitives
        self.basis: List[Dict[str, Element]] = [{}]
        self.dimension = dimension
        self.variable_symbols = sp.symbols(f"x0:{self.dimension}")

    def build_basis(self, J_Max, J_0, comp_call=True):
        basis_1d = build_basis_1d(self.primitives, J_Max, J_0)
        self.basis = (
            basis_1d
            if self.dimension == 1
            else extend_isotropic_tensor(basis_1d, self.dimension)
        )
        if comp_call:
            self._compute_callables()

    def _compute_callables(self) -> None:
        for group in self.basis:
            for elem in group.values():
                f = elem["function_sym"]
                expr = f(*self.variable_symbols) if isinstance(f, sp.Lambda) else f
                elem["function_num"] = sp.lambdify(self.variable_symbols, expr, "numpy")

    def apply(
        self, func: Callable[[Element], Element], comp_call: bool = True, **kwargs
    ) -> None:
        """Apply transformation func to all basis elements."""
        for group in self.basis:
            for key, elem in group.items():
                group[key] = func(elem, **kwargs)
        if comp_call:
            self._compute_callables()


# ------------------- temp helper  ------------------- #
def differentiate(elem: Element, axis: int = 0) -> Element:
    """
    Differentiate a single Element w.r.t. its own symbolic variable at given axis.
    Keeps the lambda arity identical to the basis dimensionality.
    """
    f = elem["function_sym"]
    if not isinstance(f, sp.Lambda):
        return elem

    args = f.args[0]
    vars_ = (args,) if isinstance(args, sp.Symbol) else tuple(args)
    var_to_diff = vars_[axis]
    expr_diff = sp.diff(f.expr, var_to_diff)
    elem["function_sym"] = sp.Lambda(vars_, expr_diff)
    elem["function_num"] = None
    return elem


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    primitives = Primitives_MinimalSupport()
    bh = BasisHandler(primitives=primitives, dimension=2)
    bh.build_basis(J_Max=4, J_0=2)

    # apply derivative to all basis elements
    bh.apply(differentiate, axis=0)

    # pick and plot one element
    elem = list(bh.basis[0].values())[1]
    f_num = elem["function_num"]

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
    ax.set_title(f"2D Basis Element, scale={elem['scale']}")
    plt.show()
