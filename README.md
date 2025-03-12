# Poisson Equation Solver

A PyTorch-based implementation of a Poisson equation solver with multiple solution methods.

## Overview

This repository provides an efficient implementation of a Poisson equation solver using PyTorch. The Poisson equation is a partial differential equation of the form:

∇²u = f

where ∇² is the Laplacian operator, u is the unknown function we want to solve for, and f is a known function.

## Features

- Multiple solution methods:
  - Conjugate Gradient (CG) - Fast and memory-efficient iterative method
  - LU Decomposition - Direct solver for smaller problems (for validation)
  - Cholesky Decomposition - Optimized for symmetric positive-definite matrices (for validation)
- GPU acceleration via PyTorch
- Visualization tools for solutions
- Support for Dirichlet boundary conditions
- Efficient sparse matrix operations

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Matplotlib (for visualization)


## Usage

visualize the unit test solution
```python
python poisson_solver.py
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.

