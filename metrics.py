import numpy as np

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer, MinimumEigenOptimizationResult



def get_aproximation_ratio (solutions: MinimumEigenOptimizationResult ,mu: np.array, sigma:np.array,q:float,budget:int):
    """Computes the aproximation ratio of a given solution of a qaoa"""
    #Compute the best result using the exact solver
    portfolio_min = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
    )
    qp_min = portfolio_min.to_quadratic_program()
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result_min = exact_eigensolver.solve(qp_min)
    #Same but to obtain the maximum energy
    portfolio_max = PortfolioOptimization(
        expected_returns=-mu, covariances=-sigma, risk_factor=q, budget=budget
    )
    qp_max = portfolio_max.to_quadratic_program()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result_max = exact_eigensolver.solve(qp_max)
    #Get the aproximation ratio
    ratio = (result_max.fval - solutions.fval)/(result_max.fval-result_min.fval) 
    return ratio