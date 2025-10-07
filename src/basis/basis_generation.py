from typing import Literal, TypedDict, List
import sympy as sp
from typing import Callable, Optional

x = sp.Symbol("x")
btype = Literal["left", "middle", "right"]


class Element(TypedDict):
    function_sym: sp.Lambda
    function_num: Optional[Callable]
    type: btype
    scale: int
    shift: int
    support: tuple[float, float]


def build_basis_1d(primitives, J_max: int, J_0: int = 2) -> List[List[Element]]:
    assert J_max >= J_0
    base: List[List[Element]] = []

    def mid_elem(f_method, j, k, supp) -> Element:
        f_lambda = f_method()
        expr = (2 ** (j / 2)) * f_lambda(2**j * x - (k - 2))
        return Element(
            function_sym=sp.Lambda(x, expr),
            function_num=None,
            type="middle",
            scale=j,
            shift=k,
            support=supp,
        )

    # scaling at level J0
    scals: List[Element] = []
    a_b, b_b = primitives.supports["phib"]

    scals.append(
        Element(
            function_sym=sp.Lambda(x, (2 ** (J_0 / 2)) * primitives.phib()(2**J_0 * x)),
            function_num=None,
            type="left",
            scale=J_0,
            shift=1,
            support=(a_b / (2**J_0), b_b / (2**J_0)),
        )
    )

    a, b = primitives.supports["phi"]
    for k in range(2, 2**J_0):
        supp = ((a + k - 2) / (2**J_0), (b + k - 2) / (2**J_0))
        scals.append(mid_elem(primitives.phi, J_0, k, supp))

    scals.append(
        Element(
            function_sym=sp.Lambda(
                x, (2 ** (J_0 / 2)) * primitives.phib()(2**J_0 * (1 - x))
            ),
            function_num=None,
            type="right",
            scale=J_0,
            shift=2**J_0,
            support=(1 - b_b / (2**J_0), 1 - a_b / (2**J_0)),
        )
    )
    base.append(scals)

    # wavelets for j = J0..J_max
    a_w, b_w = primitives.supports["psi"]
    a_wb, b_wb = primitives.supports["psib"]

    for j in range(J_0, J_max + 1):
        waves: List[Element] = []

        waves.append(
            Element(
                function_sym=sp.Lambda(x, (2 ** (j / 2)) * primitives.psib()(2**j * x)),
                function_num=None,
                type="left",
                scale=j,
                shift=1,
                support=(a_wb / (2**j), b_wb / (2**j)),
            )
        )

        for k in range(2, 2**j):
            supp = ((a_w + k - 2) / (2**j), (b_w + k - 2) / (2**j))
            waves.append(mid_elem(primitives.psi, j, k, supp))

        waves.append(
            Element(
                function_sym=sp.Lambda(
                    x, -(2 ** (j / 2)) * primitives.psib()(2**j * (1 - x))
                ),
                function_num=None,
                type="right",
                scale=j,
                shift=2**j,
                support=(1 - b_wb / (2**j), 1 - a_wb / (2**j)),
            )
        )

        base.append(waves)

    return base
