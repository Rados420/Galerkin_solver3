import sympy as sp

from typing import List
from src.primitives import Primitives, Primitives_MinimalSupport
from src.basis.basis_generation import Element, build_basis_1d

class BaseBasis(Primitives):
    def __init__(self,primitives:Primitives):
        self.primitives = primitives
        self.basis:List[List[Element]] = [[]]

    def build_basis(self,J_Max ,J_0, dimension:int):
        if dimension == 1:
            self.basis = build_basis_1d(self.primitives,J_Max,J_0)
        else:
            raise NotImplementedError

    def compute_callables(self) ->None:
        """
        Convert a basis with sympy.Lambda functions into numpy-callable versions.
        Adds a key 'function_sym' storing the original Lambda.
        """
        x = sp.Symbol("x")
        for group in self.basis:
            for elem in group:
                elem["function_num"]=sp.lambdify(x, elem["function_sym"], "numpy")

    def differentiate_basis(self) ->None:
        """
        Differentiate the symbolic part ('function_sym') of each Element in the basis.
        Returns a new basis with updated symbolic functions.
        """
        x = sp.Symbol("x")
        for group in self.basis:
            for elem in group:
                f_diff = sp.Lambda(x, sp.diff(elem["function_sym"](x), x))
                elem["function_sym"]=f_diff


class MinsupportBasis(BaseBasis):
    def __init__(self):
        super().__init__(primitives=Primitives_MinimalSupport)


# def lambdas_to_callables(basis: List[List[Element]]) -> List[List[Element]]:
#     """
#     Convert a basis with sympy.Lambda functions into numpy-callable versions.
#     Adds a key 'function_sym' storing the original Lambda.
#     """
#     x = sp.Symbol("x")
#     new_basis: List[List[Element]] = []
#
#     for group in basis:
#         new_group: List[Element] = []
#         for elem in group:
#             f_lambda = elem["function"]  # sympy.Lambda
#             f_num = sp.lambdify(x, f_lambda(x), "numpy")
#             new_elem = elem.copy()  # shallow copy of Element
#             new_elem["function_sym"] = f_lambda
#             new_elem["function"] = f_num  # numeric callable for fast use
#             new_group.append(new_elem)
#         new_basis.append(new_group)
#
#     return new_basis
#
#
# def differentiate_basis(basis: List[List[Element]]) -> List[List[Element]]:
#     """
#     Differentiate the symbolic part ('function_sym') of each Element in the basis.
#     Returns a new basis with updated symbolic functions.
#     """
#     new_basis: List[List[Element]] = []
#     for group in basis:
#         new_group = []
#         for elem in group:
#             f_sym = elem.get("function_sym", elem["function"])  # fall back if not set
#             f_diff = sp.Lambda(x, sp.diff(f_sym(x), x))
#             new_elem = elem.copy()
#             new_elem["function_sym"] = f_diff
#             new_elem["function"] = f_diff  # keep symbolic until lambdas_to_callables
#             new_group.append(new_elem)
#         new_basis.append(new_group)
#     return new_basis


if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt
    # from primitives import Primitives_MinimalSupport  # from your previous script
    mintest_basis=MinsupportBasis()
    mintest_basis.build_basis(J_Max=4,J_0=2,dimension=1)
    mintest_basis.compute_callables()
    element=mintest_basis.basis[0][2]
    f_num=element["function_num"]

    xx = np.linspace(0, 1, 300)
    plt.plot(xx, f_num(xx))
    plt.show()

    # primitives = Primitives_MinimalSupport()
    # basis = build_basis_1d(primitives=primitives, J_max=4)
    #
    # # pick one element and plot it
    # element = basis[0][1]
    # print(element)
    #
    # f_sym = element["function"]
    # f_num = sp.lambdify(x, f_sym(x), "numpy")  # numeric version
    #
    # xx = np.linspace(0, 1, 300)
    # plt.plot(xx, f_num(xx))
    # plt.show()
