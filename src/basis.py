
from typing import Callable, Literal, TypedDict, List
import numpy as np

btype = Literal["left", "middle", "right"]

class Element(TypedDict):
    function: Callable[[np.ndarray], np.ndarray]
    type: btype
    scale: int
    shift: int
    support: tuple[float, float]

base:list[list[Element]]

def build_basis_1d(primitives, level: int) -> List[List[Element]]:
    assert level >= 2
    base: List[List[Element]] = []

    def mid_elem(f, j, k, supp) -> Element:
        return {
            "function": (lambda x, f=f, j=j, k=k: (2**(j/2))*f((2**j)*x - (k - 2))),
            "type": "middle",
            "scale": j,
            "shift": k,
            "support": supp,
        }

    # --- scaling at level 'level'
    scals: List[Element] = []
    a_b, b_b = primitives.supports["phib"]
    # left boundary:  φ_{j,1}(x) = 2^{j/2} φ_b(2^j x)
    scals.append({
        "function": (lambda x, j=level: (2**(j/2))*primitives.phib((2**j)*x)),
        "type": "left",
        "scale": level,
        "shift": 1,
        "support": (a_b/(2**level), b_b/(2**level)),
    })
    # inner scalings:  φ_{j,k}(x) = 2^{j/2} φ(2^j x - k + 2), k=2..2^j-1
    a, b = primitives.supports["phi"]
    for k in range(2, 2**level):
        scals.append(mid_elem(primitives.phi, level, k,
                              ((a + k - 2)/(2**level), (b + k - 2)/(2**level))))
    # right boundary (mirror):  φ_{j,2^j}(x) = 2^{j/2} φ_b(2^j (1 - x))
    scals.append({
        "function": (lambda x, j=level: (2**(j/2))*primitives.phib((2**j)*(1 - x))),
        "type": "right",
        "scale": level,
        "shift": 2**level,
        "support": (1 - b_b/(2**level), 1 - a_b/(2**level)),
    })
    base.append(scals)

    # --- wavelets for j = 2..level
    a_w, b_w = primitives.supports["psi"]
    a_wb, b_wb = primitives.supports["psib"]
    for j in range(2, level + 1):
        waves: List[Element] = []
        # left boundary:  ψ_{j,1}(x) = 2^{j/2} ψ_b(2^j x)
        waves.append({
            "function": (lambda x, j=j: (2**(j/2))*primitives.psib((2**j)*x)),
            "type": "left",
            "scale": j,
            "shift": 1,
            "support": (a_wb/(2**j), b_wb/(2**j)),
        })
        # inner wavelets:  ψ_{j,k}(x) = 2^{j/2} ψ(2^j x - k + 2), k=2..2^j-1
        for k in range(2, 2**j):
            waves.append(mid_elem(primitives.psi, j, k,
                                  ((a_w + k - 2)/(2**j), (b_w + k - 2)/(2**j))))
        # right boundary (mirror with minus):  ψ_{j,2^j}(x) = -2^{j/2} ψ_b(2^j (1 - x))
        waves.append({
            "function": (lambda x, j=j: -(2**(j/2))*primitives.psib((2**j)*(1 - x))),
            "type": "right",
            "scale": j,
            "shift": 2**j,
            "support": (1 - b_wb/(2**j), 1 - a_wb/(2**j)),
        })
        base.append(waves)

    return base

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from primitives import Primitives_MinimalSupport
    primitives = Primitives_MinimalSupport()
    basis = build_basis_1d(primitives=primitives, level=6)
    element=basis[3][-1]
    print(element)
    f=element["function"]
    xx=np.linspace(0, 1, 300)
    plt.plot(xx,f(xx))
    plt.show()