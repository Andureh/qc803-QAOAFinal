import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals

def get_portfolio_data(tickers, start_date, end_date):
    """
    Fetches historical data and calculates expected return (mu) and covariance (sigma).
    """
    #Fetch raw data from Yahoo Finance

    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    #Calculate daily returns (percentage change)
    daily_returns = data.pct_change().dropna()
    
    print(daily_returns)
    #Calculate Mean Vector (mu) and Covariance Matrix (sigma)
    '''252 because if not, mu is the average daily returns and overall we want annual returns because it makes more sense.
    So just multiply the average daily returns by the 252 annual trading days'''
    mu = daily_returns.mean() * 252
    sigma = daily_returns.cov() * 252
    
    return mu, sigma, data

def normalize_data(mu, sigma):
    """
    Normalizes data to [0, 1] range.
    """
    mu_min = mu.min()
    mu_max = mu.max()
    mu_normalized = (mu - mu_min) / (mu_max - mu_min) #Normalization between [0,1]

    sigma_max = sigma.max().max()
    sigma_normalized = sigma / sigma_max #Normalization of sigma (the covariance matrix)
    
    return mu_normalized, sigma_normalized

def hamiltonean_creator(mu, sigma, budget, risk_factor=1.0):
    """
    THIS FUNCTION WAS JUST SOMETHING I DID TO TRY AND UNDERSTAND THE "CREATE PORTFOLIO" FUNCTION, IS LIKE A MANUAL VERSION OF THAT ONE

    1. Creates the class
    2. Adds parameters (Variables, Objective, Constraints)
    3. Calculates Hamiltonian (The Output)
    """
    
    #Create the Class
    qp = QuadraticProgram("Portfolio Optimization")
    
    #Add Parameters
    for i in range(len(mu)):
        qp.binary_var(name=f'stock_{i}')
        
    #Add the "Score" (Objective Function)
    #logic: Minimize (Risk - Return)
    #linear=Return (negative because we minimize), quadratic=Risk matrix
    qp.minimize(linear=-mu, quadratic=risk_factor * sigma)
    
    #Add the "Rule" (Constraint)
    linear_dict = {f'stock_{i}': 1 for i in range(len(mu))}
    
    qp.linear_constraint(linear=linear_dict, sense='==', rhs=budget, name='budget')

    #Calculate the Hamiltonian
    hamiltonian, offset = qp.to_ising()
    
    return hamiltonian

def create_portfolio_qp(mu, sigma, q=0.5, budget=None):
    """
    Creates a Quadratic Program for the Portfolio Optimization problem.
    
    Args:
        mu (numpy.ndarray): Expected returns vector.
        sigma (numpy.ndarray): Covariance matrix.
        q (float): Risk factor (0 = high risk/high return, 1 = low risk).
        budget (int): Number of assets to select. If None, defaults to half the assets.
        
    Returns:
        qp (QuadraticProgram): The mathematical formulation of the problem.
        penalty (float): The recommended penalty scaling factor for QUBO conversion.
    """
    num_assets = len(mu)
    
    if budget is None:
        budget = num_assets // 2
        
    # Set parameter to scale the budget penalty term
    # (Used later when converting to QUBO)
    penalty = num_assets 

    # Create the portfolio instance
    portfolio = PortfolioOptimization(
        expected_returns=mu, 
        covariances=sigma, 
        risk_factor=q, 
        budget=budget
    )
    
    # Convert to Qiskit's QuadraticProgram format
    qp = portfolio.to_quadratic_program()
    
    return qp, penalty,portfolio

def print_result(result,portfolio):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

#Test
tickers = ["AAPL", "GOOG", "MSFT", "AMZN"]
start_date = "2023-01-01"
end_date = "2024-01-01"


mu_raw, sigma_raw, prices = get_portfolio_data(tickers, start_date, end_date)
#Normalization
mu_norm, sigma_norm = normalize_data(mu_raw, sigma_raw)

'''
print("Normalized Expected Returns (mu):\n", mu_norm)
print("\nNormalized Covariance (sigma):\n", sigma_norm)
print("Prices: ", prices)

#Plot the covariance matrix

plt.imshow(sigma_norm, interpolation='nearest')
plt.colorbar()
plt.title("Normalized Covariance Matrix")
plt.show()

# 1. Create simple Dummy Data (3 assets)
# Returns (mu): Asset 0 earns 10%, Asset 1 earns 50%, Asset 2 earns 20%
mu_test = np.array([0.1, 0.5, 0.2])

# Covariance (sigma): How they move together (3x3 matrix)
# Diagonal (0.9, 0.5, 0.3) represents the risk (variance) of each asset.
sigma_test = np.array([
    [0.9, 0.1, 0.0],
    [0.1, 0.5, 0.1],
    [0.0, 0.1, 0.3]
])

# 2. Run your function
# We ask for a budget of 2 assets and a risk factor of 0.5
qp_test, penalty_test = create_portfolio_qp(mu_test, sigma_test, q=0.5, budget=2)
'''

qp_realdata, penalty,portfolio_qp = create_portfolio_qp(mu_norm,sigma_norm,q=0.5,budget=2)

# Exact Solver Implementation
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)

result = exact_eigensolver.solve(qp_realdata)

print_result(result,portfolio_qp)


# QAOA Implementation
algorithm_globals.random_seed = 1234

cobyla = COBYLA()
cobyla.set_options(maxiter=250)
qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp_realdata)

print_result(result,portfolio_qp)