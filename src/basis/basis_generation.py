from typing import TypedDict, List, Tuple, Optional, Callable, Dict
from itertools import product
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


def build_basis_1d(primitives, j_max: int, j_0: int = 2) -> List[Dict[str, Element]]:
    assert j_max >= j_0
    base: List[Dict[str, Element]] = []

    # scaling at level J0
    scals: Dict[str, Element] = {}
    a_b, b_b = primitives.supports["phib"]
    a, b = primitives.supports["phi"]

    scals[_make_id("S", j_0, 1)] = _create_element_1d(
        primitives.phib, j_0, 1, (a_b, b_b), ((0, -1),)
    )

    for k in range(2, 2**j_0):
        scals[_make_id("S", j_0, k)] = _create_element_1d(
            primitives.phi, j_0, k, (a, b), ((0, 0),)
        )

    scals[_make_id("S", j_0, 2**j_0)] = _create_element_1d(
        primitives.phib, j_0, 2**j_0, (a_b, b_b), ((0, 1),)
    )
    base.append(scals)

    # wavelets for j = J0..J_max
    a_w, b_w = primitives.supports["psi"]
    a_wb, b_wb = primitives.supports["psib"]

    for j in range(j_0, j_max + 1):
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


def extend_isotropic_tensor(
    base_1d: List[Dict[str, Element]], d: int
) -> List[Dict[str, Element]]:
    """Construct an isotropic tensor-product basis from 1D wavelet basis (supports up to d=3)."""

    if d == 1:
        return base_1d

    x_syms = sp.symbols(f"x0:{d}")
    nd_basis: List[Dict[str, Element]] = []

    for level_dict in base_1d:
        level_nd: Dict[str, Element] = {}
        keys = list(level_dict.keys())
        elems = list(level_dict.values())

        # Cartesian product over dimensions
        for combo in product(range(len(elems)), repeat=d):
            elms = [elems[i] for i in combo]

            # Combine supports, scales, and types
            scale = elms[0]["scale"]
            shifts = tuple(e["shift"][0] for e in elms)
            types = tuple(e["type"][0] for e in elms)
            support = tuple(e["support"][0] for e in elms)

            # Tensor product of symbolic functions
            f_expr = 1
            for i in range(d):
                f_expr *= elms[i]["function_sym"](x_syms[i])

            func_sym = sp.Lambda(x_syms, f_expr)

            # ID tag, short but unique
            tag = "T" + "_".join(keys[i] for i in combo)

            level_nd[tag] = Element(
                function_sym=func_sym,
                function_num=None,
                scale=scale,
                shift=shifts,
                type=types,
                support=support,
            )

        nd_basis.append(level_nd)

    return nd_basis


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    from src.primitives import Primitives_MinimalSupport

    primitives = Primitives_MinimalSupport()
    base = build_basis_1d(primitives, 4, 2)
    nd_basis = extend_isotropic_tensor(base, 2)

    elem = list(nd_basis[2].values())[0]
    d = len(elem["support"])
    x_range = np.linspace(0, 1, 200)

    if d == 1:
        x = sp.Symbol("x")
        f = sp.lambdify(x, elem["function_sym"](x), "numpy")
        plt.plot(x_range, f(x_range))
        plt.title(f"1D basis element, scale={elem['scale']}")
        plt.show()

    elif d == 2:
        x0, x1 = sp.symbols("x0 x1")
        f = sp.lambdify((x0, x1), elem["function_sym"](x0, x1), "numpy")
        X, Y = np.meshgrid(x_range, x_range)
        Z = f(X, Y)
        plt.contourf(X, Y, Z, 100, cmap="viridis")
        plt.title(f"2D basis element, scale={elem['scale']}")
        plt.colorbar()
        plt.show()

    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        x0, x1, x2 = sp.symbols("x0 x1 x2")
        f = sp.lambdify((x0, x1, x2), elem["function_sym"](x0, x1, x2), "numpy")
        X, Y = np.meshgrid(x_range, x_range)
        Z = f(X, Y, 0.5)  # slice at z=0.5
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_title(f"3D basis slice z=0.5, scale={elem['scale']}")
        plt.show()
