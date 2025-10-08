from typing import TypedDict, List, Tuple, Optional, Callable, Dict
import sympy as sp

x = sp.Symbol("x")


class Element(TypedDict):
    function_sym: sp.Lambda
    function_num: Optional[Callable]
    scale: int
    shift: Tuple[int, ...]
    type: Tuple[
        Tuple[int, int], ...
    ]  # (a,b): a=0->scaling,1->wavelet ; b=-1->left,0->mid,1->right
    support: Tuple[Tuple[float, float], ...]


def _create_element_1d(
    f_method: Callable[[], Callable],
    j: int,
    k: int,
    base_supp: Tuple[float, float],
    etype: Tuple[Tuple[int, int], ...],
) -> Element:
    """Create a 1D element, computing support and correct transform/sign from type.

    etype encoding: (a,b) with a=0 scaling, a=1 wavelet; b=-1 left, 0 middle, 1 right.
    Matches definitions in the paper: (2.6) for scaling, (2.9) for wavelets.
    """
    a, b = base_supp
    a_type, b_side = etype[0]

    # Support per side
    if b_side == -1:  # left
        supp = (a / (2**j), b / (2**j))
    elif b_side == 1:  # right
        supp = (1 - b / (2**j), 1 - a / (2**j))
    else:  # middle
        supp = ((a + k - 2) / (2**j), (b + k - 2) / (2**j))

    # Argument per side (and sign for right boundary wavelets)
    f_lambda = f_method()
    if b_side == -1:  # left boundary: f(2^j x)
        arg = 2**j * x
        sign = 1
    elif b_side == 1:  # right boundary: f(2^j (1-x)), extra minus for wavelets only
        arg = 2**j * (1 - x)
        sign = -1 if a_type == 1 else 1
    else:  # middle: f(2^j x - (k-2))
        arg = 2**j * x - (k - 2)
        sign = 1

    expr = sign * (2 ** (j / 2)) * f_lambda(arg)

    return Element(
        function_sym=sp.Lambda(x, expr),
        function_num=None,
        scale=j,
        shift=(k,),
        type=etype,
        support=(supp,),
    )


def _make_id(prefix: str, j: int, k: int) -> str:
    return f"{prefix}{j}_{k}"


def build_basis_1d(primitives, J_max: int, J_0: int = 2) -> List[Dict[str, Element]]:
    assert J_max >= J_0
    base: List[Dict[str, Element]] = []

    # scaling at level J0
    scals: Dict[str, Element] = {}
    a_b, b_b = primitives.supports["phib"]
    a, b = primitives.supports["phi"]

    scals[_make_id("S", J_0, 1)] = _create_element_1d(
        primitives.phib, J_0, 1, (a_b, b_b), ((0, -1),)
    )

    for k in range(2, 2**J_0):
        scals[_make_id("S", J_0, k)] = _create_element_1d(
            primitives.phi, J_0, k, (a, b), ((0, 0),)
        )

    scals[_make_id("S", J_0, 2**J_0)] = _create_element_1d(
        primitives.phib, J_0, 2**J_0, (a_b, b_b), ((0, 1),)
    )
    base.append(scals)

    # wavelets for j = J0..J_max
    a_w, b_w = primitives.supports["psi"]
    a_wb, b_wb = primitives.supports["psib"]

    for j in range(J_0, J_max + 1):
        waves: Dict[str, Element] = {}
        waves[_make_id("W", j, 1)] = _create_element_1d(
            primitives.psib, j, 1, (a_wb, b_wb), ((1, -1),)
        )

        for k in range(2, 2**j):
            waves[_make_id("W", j, k)] = _create_element_1d(
                primitives.psi, j, k, (a_w, b_w), ((1, 0),)
            )

        waves[_make_id("W", j, 2**j)] = _create_element_1d(
            primitives.psib, j, 2**j, (a_wb, b_wb), ((1, 1),)
        )
        base.append(waves)

    return base
