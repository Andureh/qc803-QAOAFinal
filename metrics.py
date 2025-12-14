import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer, MinimumEigenOptimizationResult
from qiskit_algorithms.utils import algorithm_globals


def get_aproximation_ratio (solutions: MinimumEigenOptimizationResult ,portfolio: PortfolioOptimization):
    """Computes the aproximation ratio of a given solution of a qaoa"""
    #Compute the best result using the exact solver
    qp = portfolio.to_quadratic_program()
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result = exact_eigensolver.solve(qp)
    #Get the aproximation ratio
    ratio = solutions.fval/result.fval
    return ratio