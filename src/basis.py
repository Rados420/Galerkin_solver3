from typing import Literal, TypedDict, List
import sympy as sp

x = sp.Symbol("x")
btype = Literal["left", "middle", "right"]


class Element(TypedDict):
    function: sp.Lambda
    type: btype
    scale: int
    shift: int
    support: tuple[float, float]


def build_basis_1d(primitives, J_max: int, J_0: int = 2) -> List[List[Element]]:
    """
    Build symbolic basis elements (Scaling + Wavelets) using sympy.Lambda.
    """
    assert J_max >= J_0
    base: List[List[Element]] = []

    def mid_elem(f_method, j, k, supp) -> Element:
        f_lambda = f_method()  # get the Lambda from primitives
        expr = (2 ** (j / 2)) * f_lambda(2**j * x - (k - 2))
        return {
            "function": sp.Lambda(x, expr),
            "type": "middle",
            "scale": j,
            "shift": k,
            "support": supp,
        }

    # --- scaling at level J0
    scals: List[Element] = []
    a_b, b_b = primitives.supports["phib"]

    # left boundary φ_b
    scals.append(
        {
            "function": sp.Lambda(x, (2 ** (J_0 / 2)) * primitives.phib()(2**J_0 * x)),
            "type": "left",
            "scale": J_0,
            "shift": 1,
            "support": (a_b / (2**J_0), b_b / (2**J_0)),
        }
    )

    # inner scalings
    a, b = primitives.supports["phi"]
    for k in range(2, 2**J_0):
        supp = ((a + k - 2) / (2**J_0), (b + k - 2) / (2**J_0))
        scals.append(mid_elem(primitives.phi, J_0, k, supp))

    # right boundary φ_b mirrored
    scals.append(
        {
            "function": sp.Lambda(
                x, (2 ** (J_0 / 2)) * primitives.phib()(2**J_0 * (1 - x))
            ),
            "type": "right",
            "scale": J_0,
            "shift": 2**J_0,
            "support": (1 - b_b / (2**J_0), 1 - a_b / (2**J_0)),
        }
    )
    base.append(scals)

    # --- wavelets for j = J0..J_max
    a_w, b_w = primitives.supports["psi"]
    a_wb, b_wb = primitives.supports["psib"]

    for j in range(J_0, J_max + 1):
        waves: List[Element] = []

        # left boundary ψ_b
        waves.append(
            {
                "function": sp.Lambda(x, (2 ** (j / 2)) * primitives.psib()(2**j * x)),
                "type": "left",
                "scale": j,
                "shift": 1,
                "support": (a_wb / (2**j), b_wb / (2**j)),
            }
        )

        # inner wavelets
        for k in range(2, 2**j):
            supp = ((a_w + k - 2) / (2**j), (b_w + k - 2) / (2**j))
            waves.append(mid_elem(primitives.psi, j, k, supp))

        # right boundary ψ_b mirrored (with minus)
        waves.append(
            {
                "function": sp.Lambda(
                    x, -(2 ** (j / 2)) * primitives.psib()(2**j * (1 - x))
                ),
                "type": "right",
                "scale": j,
                "shift": 2**j,
                "support": (1 - b_wb / (2**j), 1 - a_wb / (2**j)),
            }
        )

        base.append(waves)

    return base


def lambdas_to_callables(basis: List[List[Element]]) -> List[List[Element]]:
    """
    Convert a basis with sympy.Lambda functions into numpy-callable versions.
    Adds a key 'function_sym' storing the original Lambda.
    """
    x = sp.Symbol("x")
    new_basis: List[List[Element]] = []

    for group in basis:
        new_group: List[Element] = []
        for elem in group:
            f_lambda = elem["function"]  # sympy.Lambda
            f_num = sp.lambdify(x, f_lambda(x), "numpy")
            new_elem = elem.copy()  # shallow copy of Element
            new_elem["function_sym"] = f_lambda
            new_elem["function"] = f_num  # numeric callable for fast use
            new_group.append(new_elem)
        new_basis.append(new_group)

    return new_basis


def differentiate_basis(basis: List[List[Element]]) -> List[List[Element]]:
    """
    Differentiate the symbolic part ('function_sym') of each Element in the basis.
    Returns a new basis with updated symbolic functions.
    """
    new_basis: List[List[Element]] = []
    for group in basis:
        new_group = []
        for elem in group:
            f_sym = elem.get("function_sym", elem["function"])  # fall back if not set
            f_diff = sp.Lambda(x, sp.diff(f_sym(x), x))
            new_elem = elem.copy()
            new_elem["function_sym"] = f_diff
            new_elem["function"] = f_diff  # keep symbolic until lambdas_to_callables
            new_group.append(new_elem)
        new_basis.append(new_group)
    return new_basis


if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt
    from primitives import Primitives_MinimalSupport  # from your previous script

    primitives = Primitives_MinimalSupport()
    basis = build_basis_1d(primitives=primitives, J_max=4)

    # pick one element and plot it
    element = basis[0][1]
    print(element)

    f_sym = element["function"]
    f_num = sp.lambdify(x, f_sym(x), "numpy")  # numeric version

    xx = np.linspace(0, 1, 300)
    plt.plot(xx, f_num(xx))
    plt.show()
