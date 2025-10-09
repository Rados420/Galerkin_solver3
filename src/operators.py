from src.basis.basis_generation import Element
import sympy as sp


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
