from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List
import numpy as np


@dataclass
class ScalingFamily:
    """Abstract base for scaling functions."""
    supports: Dict[str, Tuple[float,float]] = field(default_factory=dict)

    def phi(self, x): raise NotImplementedError
    def phib(self, x): raise NotImplementedError
    def dphi(self, x): raise NotImplementedError
    def dphib(self, x): raise NotImplementedError

    def as_dict(self) -> Dict[str, Callable]:
        return {
            "phi": self.phi, "phib": self.phib,
            "dphi": self.dphi, "dphib": self.dphib
        }

@dataclass
class Primitives(ScalingFamily):
    """Generic construction of wavelets from a scaling family."""
    # specs: lists of (coeff, base, shift)
    psi_spec: List[Tuple[float,str,float]] = field(default_factory=list)
    psib_spec: List[Tuple[float,str,float]] = field(default_factory=list)
    supports: Dict[str, Tuple[float,float]] = field(default_factory=dict)

    def __post_init__(self):
        # compute supports for wavelets based on scaling supports + spec
        def compute_support(spec):
            supps = []
            for coeff, base, shift in spec:
                a,b = self.supports[base]
                supps.append(((a+shift)/2, (b+shift)/2))
            a = min(s[0] for s in supps)
            b = max(s[1] for s in supps)
            return (a,b)

        if self.psi_spec:
            self.supports["psi"]  = compute_support(self.psi_spec)
        if self.psib_spec:
            self.supports["psib"] = compute_support(self.psib_spec)

    def psi(self,x):
        return sum(c * getattr(self,base)(2*x - sh) for c,base,sh in self.psi_spec)
    def psib(self,x):
        return sum(c * getattr(self,base)(2*x - sh) for c,base,sh in self.psib_spec)

    def dpsi(self,x):
        return sum(c*2*getattr(self,"d"+base)(2*x - sh) for c,base,sh in self.psi_spec)
    def dpsib(self,x):
        return sum(c*2*getattr(self,"d"+base)(2*x - sh) for c,base,sh in self.psib_spec)


    def as_dict(self):
        d = super().as_dict()
        d.update({
            "psi": self.psi, "psib": self.psib,
            "dpsi": self.dpsi, "dpsib": self.dpsib
        })
        return d

class SF_MinimalSupport(ScalingFamily):
    supports = {
        "phi":  (0,3),
        "phib": (0,2)
    }
    def phi(self,x):
        x=np.asarray(x); y=np.zeros_like(x,float)
        m=(0<=x)&(x<=1); y[m]=0.5*x[m]**2
        m=(1<x)&(x<=2); y[m]=-x[m]**2+3*x[m]-1.5
        m=(2<x)&(x<=3); y[m]=0.5*x[m]**2-3*x[m]+4.5
        return y
    def phib(self,x):
        x=np.asarray(x); y=np.zeros_like(x,float)
        m=(0<=x)&(x<=1); y[m]=-9/4*x[m]**2+3*x[m]
        m=(1<x)&(x<=2); y[m]=3/4*x[m]**2-3*x[m]+3
        return y
    def dphi(self,x):
        x=np.asarray(x); y=np.zeros_like(x,float)
        m=(0<=x)&(x<=1); y[m]=x[m]
        m=(1<x)&(x<=2); y[m]=-2*x[m]+3
        m=(2<x)&(x<=3); y[m]=x[m]-3
        return y
    def dphib(self,x):
        x=np.asarray(x); y=np.zeros_like(x,float)
        m=(0<=x)&(x<=1); y[m]=-9/2*x[m]+3
        m=(1<x)&(x<=2); y[m]=1.5*x[m]-3
        return y

class Primitives_MinimalSupport(SF_MinimalSupport, Primitives):
    def __init__(self):
        # define wavelet as -0.5 φ(2x-1) + 0.5 φ(2x-2)
        psi_spec  = [( -0.5, "phi", 1), (0.5, "phi", 2)]
        psib_spec = [( -0.5, "phib", 0), (0.5, "phi", 0)]
        super().__init__(psi_spec=psi_spec, psib_spec=psib_spec,
                         supports=self.supports.copy())


if __name__ == "__main__":
    P = Primitives_MinimalSupport()
    print("Supports:", P.supports)
    print("ψ support:", P.supports["psi"])
    print("ψb support:", P.supports["psib"])

    xs = np.linspace(0,3,200)
    import matplotlib.pyplot as plt
    plt.plot(xs,P.phi(xs),label="phi")
    plt.plot(xs,P.phib(xs),label="phib")
    plt.plot(xs,P.psi(xs),label="psi")
    plt.plot(xs,P.psib(xs),label="psib")
    plt.legend(); plt.show()