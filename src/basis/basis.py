import sympy as sp
import numpy as np
from typing import List, Dict, Callable
from src.primitives import Primitives, Primitives_MinimalSupport
from src.basis.basis_generation import Element, build_basis_1d, extend_isotropic_tensor
from numpy.polynomial.legendre import leggauss


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

    def flatten(self) -> List[Element]:
        """Flatten basis element into a single list."""
        flattened = [el for group in self.basis for el in group.values()]
        return flattened

    @staticmethod
    def _finest_level_from_basis(basis1d):
        levels = [e.get("scale", None) for e in basis1d]  # 'scale' is in your Element
        levels = [L for L in levels if L is not None]
        return max(levels) if levels else 8

    def _gauss_on_dyadic(self, level: int, q: int):
        """
        Build tensorable 1D Gauss–Legendre nodes and weights on a dyadic mesh of size m=2**level.
        Returns concatenated nodes x in (0,1) and weights w of length m*q.
        """
        m = 2**level
        xi, wi = leggauss(q)
        h = 1.0 / m
        a = np.arange(m) * h
        b = a + h
        mid = (a + b)[:, None] * 0.5
        rad = (b - a)[:, None] * 0.5
        X = mid + rad * xi[None, :]
        W = rad * wi[None, :]
        return X.reshape(-1), W.reshape(-1)

    def _eval_basis_on_nodes(self, basis1d, nodes: np.ndarray) -> np.ndarray:
        """
        Φ[i,j] = φ_j(x_i) for 1D basis.
        Uses the first interval of Element['support'] which is a tuple of tuples for general-d.
        """
        n_nodes = nodes.size
        n_funcs = len(basis1d)
        Ph = np.zeros((n_nodes, n_funcs))

        for j, e in enumerate(basis1d):
            f = e["function_num"]
            if f is None:
                raise ValueError(
                    "Element['function_num'] is None; provide a numeric callable."
                )
            supp = e.get("support", None)
            if not supp or len(supp) == 0:
                # If support missing, assume whole [0,1]
                a, b = 0.0, 1.0
            else:
                # 1D: take the first interval (a,b) from the tuple of tuples
                a, b = supp[0]
            mask = (nodes >= a) & (nodes <= b)
            if np.any(mask):
                Ph[mask, j] = f(nodes[mask])
            # outside support stays zero
        return Ph

    def project_rhs_1d_gauss(self, f, basis1d=None, level=None, q=5):
        """
        High-accuracy 1D RHS projection using dyadic Gauss–Legendre quadrature.
        Computes b_i = ∫ f(x) φ_i(x) dx for all basis functions φ_i.
        """
        if basis1d is None:
            basis1d = self.flatten()

        if level is None:
            level = self._finest_level_from_basis(basis1d) + 1

        xs, wx = self._gauss_on_dyadic(level, q)
        Phix = self._eval_basis_on_nodes(basis1d, xs)  # shape (nx, N)
        F = f(xs)
        b = Phix.T @ (F * wx)
        return b

    def project_rhs_2d_gauss(self, f, level=None, basis1d=None, qx=5, qy=5):
        """
        More precise yet fast projection: tensor-product Gauss–Legendre on a dyadic mesh.
        Returns the vectorized RHS in Kronecker ordering (length N1*N1).
        """
        if basis1d is None:
            basis1d = self.flatten()

        if level is None:
            level = (
                self._finest_level_from_basis(basis1d) + 1
            )  # a bit finer than the basis
        print(level)
        xs, wx = self._gauss_on_dyadic(level, qx)
        ys, wy = self._gauss_on_dyadic(level, qy)

        Phix = self._eval_basis_on_nodes(basis1d, xs)  # (nx, N1)
        Phiy = self._eval_basis_on_nodes(basis1d, ys)  # (ny, N1)

        # Evaluate f on tensor grid and weight it
        F = np.empty((xs.size, ys.size))
        for i, x in enumerate(xs):
            F[i, :] = f(x, ys)
        F *= wx[:, None] * wy[None, :]

        # Tensor quadrature: (N1 x nx) @ (nx x ny) @ (ny x N1) -> (N1 x N1)
        B = Phix.T @ F @ Phiy
        return B.ravel(order="C")

    def evaluate_solution(self, coeffs, xs, basis1d=None):
        """Evaluate linear combination of basis functions at points xs."""
        if basis1d is None:
            basis1d = self.flatten()
        return np.array(
            [
                sum(c * e["function_num"](xx) for c, e in zip(coeffs, basis1d))
                for xx in xs
            ]
        )

    def evaluate_solution_2d_kron(self, coeffs_kron, xs, ys, basis1d=None):
        """Evaluate u(x,y) = Σ_ij C[i,j] φ_i(x)φ_j(y)."""
        if basis1d is None:
            basis1d = self.flatten()
        N1 = len(basis1d)
        C = coeffs_kron.reshape(N1, N1)
        Phix = np.array([[e["function_num"](x) for e in basis1d] for x in xs])
        Phiy = np.array([[e["function_num"](y) for e in basis1d] for y in ys])
        return Phix @ C @ Phiy.T


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from src.operators import differentiate

    primitives = Primitives_MinimalSupport()
    bh = BasisHandler(primitives=primitives, dimension=1)
    bh.build_basis(J_Max=4, J_0=2)

    # apply derivative to all basis elements
    bh.apply(differentiate, axis=0)

    # pick and plot one element
    elem = list(bh.basis[0].values())[1]
    f_num = elem["function_num"]

    # mesh grid for plotting
    xx = np.linspace(0, 1, 200)
    plt.plot(xx, f_num(xx))
    plt.show()
