from qiskit_aer.primitives import Sampler
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from metrics import get_aproximation_ratio
from normalize_data import normalize_data

import datetime
import argparse

def run_qaoa(num_assets, q, budget, penalty, p):
    
    if penalty is None:
        penalty = num_assets
    seed = 1234

    # Generate expected return and covariance matrix from (random) time-series
    stocks = [("TICKER%s" % i) for i in range(num_assets)]
    data = RandomDataProvider(
        tickers=stocks,
        start=datetime.datetime(2016, 1, 1),
        end=datetime.datetime(2016, 1, 30),
        seed=seed,
    )
    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()
    mu_norm, sigma_norm = normalize_data(mu,sigma)

    portfolio = PortfolioOptimization(
        expected_returns=mu_norm, covariances=sigma_norm, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()

    algorithm_globals.random_seed = 1234

    cobyla = COBYLA()
    cobyla.set_options(maxiter=100)         
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=p)
    qaoa = MinimumEigenOptimizer(qaoa_mes,penalty)
    result = qaoa.solve(qp)

    
    #print_result(result,portfolio)
    arat = get_aproximation_ratio(result,mu_norm,sigma_norm,q,budget)
    return arat
    


def main():
    parser = argparse.ArgumentParser(description="Run QAOA portfolio optimization")

    parser.add_argument("--n", type=int, default=10,
                        help="Number of assets (qubits)")
    parser.add_argument("--q", type=float, default=0.5,
                        help="Risk factor")
    parser.add_argument("--budget", type=int, default=2,
                        help="Budget constraint")
    parser.add_argument("--penalty", type=float, default=None,
                        help="restriction penalty (lambda)")
    parser.add_argument("--p", type=int, default=3,
                        help="Number of reps for the qaoa")

    args = parser.parse_args()

    run_qaoa(
        num_assets=args.n,
        q=args.q,
        budget=args.budget,
        penalty=args.penalty,
        p=args.p
    )

if __name__ == "__main__":
    main()



