from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import sympy as sp

x = sp.Symbol("x")


@dataclass
class ScalingFamily:
    """Abstract base for scaling functions (symbolic)."""

    supports: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def phi(self):
        raise NotImplementedError

    def phib(self):
        raise NotImplementedError

    def get_primitives(self) -> Dict[str, sp.Lambda]:
        return {"phi": self.phi(), "phib": self.phib()}


@dataclass
class Primitives(ScalingFamily):
    psi_spec: List[Tuple[float, str, float]] = field(default_factory=list)
    psib_spec: List[Tuple[float, str, float]] = field(default_factory=list)

    def __post_init__(self):
        def compute_support(spec):
            supps = []
            for coeff, base, shift in spec:
                a, b = self.supports[base]
                supps.append(((a + shift) / 2, (b + shift) / 2))
            return (min(s[0] for s in supps), max(s[1] for s in supps))

        if self.psi_spec:
            self.supports["psi"] = compute_support(self.psi_spec)
        if self.psib_spec:
            self.supports["psib"] = compute_support(self.psib_spec)

    def psi(self):
        return sp.Lambda(
            x,
            sum(c * getattr(self, base)()(2 * x - sh) for c, base, sh in self.psi_spec),
        )

    def psib(self):
        return sp.Lambda(
            x,
            sum(
                c * getattr(self, base)()(2 * x - sh) for c, base, sh in self.psib_spec
            ),
        )

    def get_primitives(self):
        d = super().get_primitives()
        d.update({"psi": self.psi(), "psib": self.psib()})
        return d

    def get_supports(self):
        return self.supports


class SF_MinimalSupport(ScalingFamily):
    supports = {"phi": (0, 3), "phib": (0, 2)}

    def phi(self):
        return sp.Lambda(
            x,
            sp.Piecewise(
                (x**2 / 2, (x >= 0) & (x <= 1)),
                (-(x**2) + 3 * x - sp.Rational(3, 2), (x > 1) & (x <= 2)),
                (x**2 / 2 - 3 * x + sp.Rational(9, 2), (x > 2) & (x <= 3)),
                (0, True),
            ),
        )

    def phib(self):
        return sp.Lambda(
            x,
            sp.Piecewise(
                (-sp.Rational(9, 4) * x**2 + 3 * x, (x >= 0) & (x <= 1)),
                (sp.Rational(3, 4) * x**2 - 3 * x + 3, (x > 1) & (x <= 2)),
                (0, True),
            ),
        )


class Primitives_MinimalSupport(SF_MinimalSupport, Primitives):
    def __init__(self):
        psi_spec = [(-0.5, "phi", 1), (0.5, "phi", 2)]
        psib_spec = [(-0.5, "phib", 0), (0.5, "phi", 0)]
        super().__init__(
            psi_spec=psi_spec, psib_spec=psib_spec, supports=self.supports.copy()
        )


if __name__ == "__main__":
    P = Primitives_MinimalSupport()
    funcs = P.get_primitives()

    print("Supports:", P.supports)
    print("ψ symbolic:", funcs["psi"].expr)
    print("ψb symbolic:", funcs["psib"].expr)

    # Convert symbolic Lambda -> numerical fast callables
    psi_num = sp.lambdify(x, funcs["psi"](x), "numpy")
    psib_num = sp.lambdify(x, funcs["psib"](x), "numpy")

    import numpy as np, matplotlib.pyplot as plt

    xs = np.linspace(0, 3, 400)
    plt.plot(xs, sp.lambdify(x, funcs["phi"](x), "numpy")(xs), label="phi")
    plt.plot(xs, sp.lambdify(x, funcs["phib"](x), "numpy")(xs), label="phib")
    plt.plot(xs, psi_num(xs), label="psi")
    plt.plot(xs, psib_num(xs), label="psib")
    plt.legend()
    plt.show()
