import numpy as np
from typing import Callable, Dict
from primitives import Primitives


def make_scaling(f: Callable, j: int, k: int, kind: str) -> Callable:
    """Return φ_{j,k} or φb-based callable depending on 'kind'."""
    if kind == "phi":
        return lambda x: 2 ** (j / 2) * f(2 ** j * x - k + 2)
    elif kind == "phib_left":
        return lambda x: 2 ** (j / 2) * f(2 ** j * x)
    elif kind == "phib_right":
        return lambda x: 2 ** (j / 2) * f(2 ** j * (1 - x))
    else:
        raise ValueError(f"Unknown scaling kind {kind}")


def make_wavelet(f: Callable, j: int, k: int, kind: str) -> Callable:
    """Return ψ_{j,k} or ψb-based callable depending on 'kind'."""
    if kind == "psi":
        return lambda x: 2 ** (j / 2) * f(2 ** j * x - k + 2)
    elif kind == "psib_left":
        return lambda x: 2 ** (j / 2) * f(2 ** j * x)
    elif kind == "psib_right":
        return lambda x: -2 ** (j / 2) * f(2 ** j * (1 - x))
    else:
        raise ValueError(f"Unknown wavelet kind {kind}")


def build_basis(primitives:Primitives, jmax: int) -> Dict[str, Callable]:
    """
    Construct basis functions up to level jmax

    Parameters
    ----------
    primitives : object with attributes phi, phib, psi, psib
    jmax : int, maximum level

    Returns
    -------
    Dict[str, Callable] : identifier → function
    """
    basis = {}

    # Scaling level j=2
    j = 2
    n = 2 ** j
    basis.update({
        f"phi_{j},{k}": make_scaling(primitives.phi, j, k, "phi")
        for k in range(2, n)
    })
    basis[f"phi_{j},1"] = make_scaling(primitives.phib, j, 1, "phib_left")
    basis[f"phi_{j},{n}"] = make_scaling(primitives.phib, j, n, "phib_right")

    # Wavelet levels j=2..jmax
    for j in range(2, jmax + 1):
        n = 2 ** j
        basis.update({
            f"psi_{j},{k}": make_wavelet(primitives.psi, j, k, "psi")
            for k in range(2, n)
        })
        basis[f"psi_{j},1"] = make_wavelet(primitives.psib, j, 1, "psib_left")
        basis[f"psi_{j},{n}"] = make_wavelet(primitives.psib, j, n, "psib_right")

    return basis


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from primitives import Primitives_MinimalSupport
    primitives = Primitives_MinimalSupport()
    basis = build_basis(primitives=primitives, jmax=3)
    # print(basis.keys())
    xx=np.linspace(0, 1, 300)
    f=basis["phi_2,2"]
    plt.plot(xx,f(xx))
    plt.show()