# QAOA Portfolio Optimization

This project explores the application of the Quantum Approximate Optimization Algorithm (QAOA) to the Portfolio Optimization problem. We analyze the performance of various classical optimizers (COBYLA, SPSA, POWELL) and investigate the impact of noise, circuit depth, and penalty factors on solution quality.

This repository is structured as a Python package to support the analysis performed in `final_report.ipynb`.

## Project Structure

```text
project_repo/
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # Project documentation
├── src/
│   └── qc803_project/       # Source code package
│       ├── __init__.py
│       └── project.ipynb     # Core logic (Data fetching, QAOA, Solvers)
└── QC803FinalReport.pdf      # Main analysis and results
```

# Installation
1. Prerequisite
Ensure you have Python 3.11 installed.

## Dependencies
qiskit & qiskit-algorithms

qiskit-optimization & qiskit-finance

yfinance (Real market data)

matplotlib & pandas

The dependecies used can be installed by:
pip install qiskit==1.0.1 qiskit-finance==0.4.1 qiskit-aer==0.13.3 qiskit-algorithms==0.3.0 qiskit-optimization==0.6.1 matplotlib

# Usage
The entire analysis is contained within the Jupyter Notebook (project.ipynb).

# Key Experiments
The notebook covers the following analyses:

QAOA vs Greedy Algorithm: Benchmarking COBYLA, SPSA, and POWELL vs the Greedy Algorithm

Optimizer Analysis: Benchmarking COBYLA, SPSA, and POWELL vs the Greedy Algorithm

Depth Analysis: Observing the effect of the Depth in the QAOA algorithm

Penalty Analysis: Analyzing the impact of constraint penalties on approximation ratios.

Noise Analysis: Testing robustness against depolarizing noise.

# Authors
André Gomes

Martín Marcuello
