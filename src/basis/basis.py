import sympy as sp

from typing import List, Dict
from src.primitives import Primitives, Primitives_MinimalSupport
from src.basis.basis_generation import Element, build_basis_1d


class BaseBasis(Primitives):
    def __init__(self, primitives: Primitives):
        self.primitives = primitives
        self.basis: List[Dict[Element]] = [{}]

    def build_basis(self, J_Max, J_0, dimension: int):
        if dimension == 1:
            self.basis = build_basis_1d(self.primitives, J_Max, J_0)
        else:
            raise NotImplementedError

    def compute_callables(self) -> None:
        """
        Convert a basis with sympy.Lambda functions into numpy-callable versions.
        """
        x = sp.Symbol("x")
        for group in self.basis:
            for elem in group.values():
                f = elem["function_sym"]
                # --- minimal fix: pass a plain Expr to lambdify
                expr = f(x) if isinstance(f, sp.Lambda) else f
                elem["function_num"] = sp.lambdify(x, expr, "numpy")

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
    import numpy as np, matplotlib.pyplot as plt

    primitives = Primitives_MinimalSupport()
    mintest_basis = BaseBasis(primitives=primitives)
    mintest_basis.build_basis(J_Max=4, J_0=2, dimension=1)
    mintest_basis.compute_callables()
    element = list(mintest_basis.basis[1].values())[3]
    print(element)
    f_num = element["function_num"]

    xx = np.linspace(0, 1, 300)
    plt.plot(xx, f_num(xx))
    plt.show()
