# Wavelet Galerkin Solver

- Updated (10. 10. 2025)

A compact, extensible Python framework for solving PDEs using **wavelet-based Galerkin methods**.  
Currently in development.

---

### 📁 Project Structure 

```text
src/
├── basis/                      # Basis generation and manipulation
│   ├── basis.py                # BasisHandler class (builds 1D and multi-D bases)
│   ├── basis_generation.py
│   └── primitives.py           # Definitions of primitive scaling and wavelet functions
│
├── matrix_generation.py        # Mass / stiffness assembly and tensor extensions
├── operators.py                # Differentiation operators acting on basis
├── plotting.py                 # 1D and 2D plotting and animation utilities
└── ...

demo_runs/
├── demo_poisson_1d.py          # 1D Poisson solver (Galerkin)
├── demo_poisson_2d.py          # 2D Poisson solver
├── demo_wave_1d.py             # 1D Wave equation solver
└── demo_wave_2d.py             # 2D Wave equation solver
```

## 🧩 Dependencies

All dependencies are standard scientific Python packages:

```text
matplotlib==3.10.6
pip==25.1.1
scipy==1.16.2
sympy==1.14.0
```