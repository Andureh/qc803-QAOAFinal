# QAOA Portfolio Optimization

This project explores the application of the Quantum Approximate Optimization Algorithm (QAOA) to the Portfolio Optimization problem. We analyze the performance of various classical optimizers (COBYLA, SPSA, POWELL) and investigate the impact of noise, circuit depth, and penalty factors on solution quality.

This repository is structured as a Python package to support the analysis performed in `project.ipynb`.

## 📂 Project Structure

```text
project_repo/
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # Project documentation
├── src/
│   └── qc803_project/       # Source code package
│       ├── __init__.py
│       └── project.ipynb     # Core logic (Data fetching, QAOA, Solvers)
└── final_report.pdf       # Main analysis and results (Jupyter Notebook)

Installation
To reproduce the results, follow these steps to set up the environment.

1. Prerequisite
Ensure you have Python 3.11 installed.

Usage
The entire analysis is contained within the Jupyter Notebook.

Key Experiments
The notebook covers the following analyses:

Optimizer Comparison: Benchmarking COBYLA, SPSA, and POWELL.

Noise Analysis: Testing robustness against depolarizing noise.

Penalty Analysis: Analyzing the impact of constraint penalties on approximation ratios.

Dependencies
qiskit & qiskit-algorithms

qiskit-optimization & qiskit-finance

yfinance (Real market data)

matplotlib & pandas

To install the necessary dependencies just install the following:
pip install qiskit==1.0.1 qiskit-finance==0.4.1 qiskit-aer==0.13.3 qiskit-algorithms==0.3.0 qiskit-optimization==0.6.1 matplotlib

Authors
[André Gomes]

[Teammate 1 Name]
