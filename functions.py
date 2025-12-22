import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA, POWELL
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.noise import NoiseModel, depolarizing_error

def get_famous_tickers(n):
    """
    Returns 'n' tickers from a predefined pool. 
    The selection is randomized but deterministic (always returns the same set for the same n).
    """
    # 1. Define your large pool of stocks
    ticker_pool = [
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX",  # Tech
        "JPM", "BAC", "V", "MA", "GS",                                   # Finance
        "WMT", "COST", "PG", "KO", "PEP", "MCD",                         # Consumer
        "JNJ", "PFE", "MRK", "UNH",                                      # Healthcare
        "DIS", "ADBE", "CRM", "INTC", "AMD", "IBM", "ORCL"               # Others
    ]
    
    # Check if we have enough stocks
    if n > len(ticker_pool):
        raise ValueError(f"Requesting {n} tickers, but pool only has {len(ticker_pool)}.")

    # 2. Create a specific random number generator with a FIXED seed
    # We use a local 'rng' object so we don't mess up your global random.seed() elsewhere
    rng = random.Random(42) 
    
    # 3. Shuffle the entire pool deterministically
    # Because the seed is always 42, the order will be identical every time you run this.
    shuffled_pool = ticker_pool[:] # Make a copy so we don't modify the original list
    rng.shuffle(shuffled_pool)
    
    # 4. Return the top n
    return shuffled_pool[:n]

def get_portfolio_data(tickers, start_date, end_date, num_assets):
    """
    Fetches historical data and calculates expected return (mu) and covariance (sigma).
    """
    
    if tickers == []:
        #Fetch random data from qiskit finance
        seed = 1234
        tickers = [("TICKER%s" % i) for i in range(num_assets)]
        data = RandomDataProvider(
        tickers=tickers,
        start=datetime.datetime(2016, 1, 1),
        end=datetime.datetime(2016, 1, 30),
        seed=seed,
        )
        data.run()
        mu = data.get_period_return_mean_vector()
        sigma = data.get_period_return_covariance_matrix()
    else:
        #Fetch raw data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        #Calculate daily returns (percentage change)
        daily_returns = data.pct_change().dropna()
        
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
    
    mu_arr = mu.values if hasattr(mu, 'values') else mu
    sigma_arr = sigma.values if hasattr(sigma, 'values') else sigma

    if budget is None:
        budget = num_assets // 2
    
    # Create the portfolio instance
    portfolio = PortfolioOptimization(
        expected_returns=mu_arr, 
        covariances=sigma_arr, 
        risk_factor=q, 
        budget=budget
    )
    
    # Convert to Qiskit's QuadraticProgram format
    qp = portfolio.to_quadratic_program()
    
    return qp, portfolio

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

def get_aproximation_ratio(solution_result, mu, sigma, q, budget,greedy=False, energy = None):
    """
    Computes the approximation ratio: (C_max - C_obj) / (C_max - C_min)
    """
    
    # Check if inputs are Pandas types (have a .values attribute) and convert if so
    mu_arr = mu.values if hasattr(mu, 'values') else mu
    sigma_arr = sigma.values if hasattr(sigma, 'values') else sigma

    # 1. Compute the TRUE MINIMUM (Best possible value)
    portfolio_min = PortfolioOptimization(
        expected_returns=mu_arr,
        covariances=sigma_arr,    
        risk_factor=q, 
        budget=budget
    )
    qp_min = portfolio_min.to_quadratic_program()
    exact_mes = NumPyMinimumEigensolver()
    exact_optimizer = MinimumEigenOptimizer(exact_mes)
    result_min = exact_optimizer.solve(qp_min)
    c_min = result_min.fval

    # 2. Compute the TRUE MAXIMUM (Worst possible value)
    # Note: We negate the numpy arrays, which works perfectly
    portfolio_max = PortfolioOptimization(
        expected_returns=-mu_arr, 
        covariances=-sigma_arr, 
        risk_factor=q, 
        budget=budget
    )
    qp_max = portfolio_max.to_quadratic_program()
    result_max = exact_optimizer.solve(qp_max)
    c_max = -result_max.fval 

    # 3. Get the QAOA Value
    if greedy and energy is None:
        raise ValueError("For greedy=True, energy must be provided.")
    if greedy:
        c_obj = energy
    else:
        c_obj = solution_result.fval

    # 4. Calculate Ratio
    denominator = c_max - c_min
    if abs(denominator) < 1e-9:
        return 0.0
        
    ratio = (c_max - c_obj) / denominator
    return ratio

def run_qaoa_experiments(qp, penalty, reps, optimizers_to_test, mu, sigma, q, budget):
    """
    Runs QAOA experiments using multiple classical optimizers and calculates performance metrics.

    This function iterates through the provided optimizers, solving the given Portfolio
    Optimization problem (QP) with the QAOA algorithm. It returns a dictionary containing
    probabilities, the optimal bitstring found, the objective value, the calculated
    approximation ratio, AND the probability of finding the exact optimal solution for each run.

    Args:
        qp (QuadraticProgram): The portfolio optimization problem to solve.
        penalty (float): The penalty factor used for constraint enforcement.
        reps (int): The number of QAOA circuit layers (depth p).
        optimizers_to_test (dict): Dictionary mapping optimizer names (str) to Qiskit optimizer instances.
        mu (pd.Series or np.ndarray): Expected asset returns, used for benchmarking.
        sigma (pd.DataFrame or np.ndarray): Asset covariance matrix, used for benchmarking.
        q (float): Risk factor used in the portfolio problem formulation.
        budget (int): Budget constraint (number of assets to select).

    Returns:
        dict: A nested dictionary where keys are optimizer names and values are dictionaries containing:
            - "probs" (dict): Measurement probabilities sorted by value.
            - "optimal_str" (str): The bitstring of the optimal solution found by the optimizer.
            - "optimal_val" (float): The energy (objective value) of the solution.
            - "ratio" (float): The approximation ratio comparing the result to the true optimal.
            - "success_prob" (float): The specific probability of measuring the TRUE global optimal bitstring.
    """
    results_data = {}
    
    # We need the true optimal bitstring to check its probability in the QAOA results.
    exact_mes = NumPyMinimumEigensolver()
    exact_opt = MinimumEigenOptimizer(exact_mes)
    result_min = exact_opt.solve(qp)
    # Convert numpy array [1. 0. 1. 0.] to string "1010"
    true_optimal_str = "".join(str(int(x)) for x in result_min.x)
    
    print(f"Running QAOA experiments with reps={reps}...")
    print(f"True Optimal Target: {true_optimal_str}")
    print(f"{'Optimizer':<10} | {'Ratio C(Z)':<8} | {'Value':<10} | {'Best String':<10} | {'Success Prob':<12}")
    print("-" * 70)
    

    for label, optimizer in optimizers_to_test.items():
        # Reset seed for fairness (Global + Sampler)
        algorithm_globals.random_seed = 1234
        
        sampler = Sampler(backend_options={"seed_simulator": 1234})

        qaoa_mes = QAOA(
            sampler=sampler, 
            optimizer=optimizer, 
            reps=reps,
        )
        qaoa = MinimumEigenOptimizer(qaoa_mes, penalty=penalty)
        result = qaoa.solve(qp)
        
        # Calculate Approximation Ratio
        ratio = get_aproximation_ratio(result, mu, sigma, q, budget)
        
        # Capture optimal string found by THIS optimizer run
        found_optimal_str = "".join(str(int(x)) for x in result.x)
        
        # Extract Probabilities
        eigenstate = result.min_eigen_solver_result.eigenstate
        probabilities = (
            eigenstate.binary_probabilities()
            if hasattr(eigenstate, 'binary_probabilities')
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        )
        
        # Qiskit results can sometimes be Little Endian (reversed). Check both just in case.
        success_prob = probabilities.get(true_optimal_str, 0.0)
        if success_prob == 0.0:
            success_prob = probabilities.get(true_optimal_str[::-1], 0.0)

        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        
        # Store Data
        results_data[label] = {
            "probs": sorted_probs,
            "optimal_str": found_optimal_str,
            "optimal_val": result.fval,
            "ratio": ratio,
            "success_prob": success_prob
        }
        
        # Print nicely formatted row
        print(f"{label:<10} | {ratio:.4f}   | {result.fval:.4f}     | {found_optimal_str:<11} | {success_prob:.4f}")
        
    return results_data

def plot_qaoa_results(results_data):
    """
    Plots histograms from the data dictionary returned by run_qaoa_experiments.
    Highlights the optimal solution found in each run in GREEN.
    """
    num_plots = len(results_data)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, 5 * num_plots))
    
    # Handle single plot case (axes is not a list if nrows=1)
    if num_plots == 1:
        axes = [axes]
    
    for i, (label, data) in enumerate(results_data.items()):
        ax = axes[i]
        probs = data["probs"]
        run_optimal_str = data["optimal_str"]
        
        # Plot top 15 only
        top_n = 15
        bitstrings_raw = list(probs.keys())[:top_n]
        values = list(probs.values())[:top_n]
        
        # Reverse strings for plotting: Qiskit '0011' -> Label '1100'
        labels = [b[::-1] for b in bitstrings_raw]
        
        bars = ax.bar(labels, values, color='skyblue', edgecolor='black')
        
        # Highlight Logic
        found_in_top_n = False
        for bar, plot_label in zip(bars, labels):
            if plot_label == run_optimal_str:
                bar.set_facecolor('green')
                bar.set_linewidth(2)
                found_in_top_n = True
                
                # Add "Optimal" text above bar
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        'Optimal', ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

        # Formatting
        ax.set_title(f"Scenario: {label} | Found Optimal: {run_optimal_str}", fontsize=14)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if not found_in_top_n:
            ax.text(0.95, 0.95, f"Optimal ({run_optimal_str}) not in top {top_n}!", 
                    transform=ax.transAxes, color='red', ha='right', fontweight='bold')

    plt.xlabel("Bitstrings (Selections)", fontsize=12)
    plt.tight_layout()
    plt.show()

def solve_greedy(mu, sigma, budget, qp):
    """
    Implements a greedy heuristic that selects the top 'budget' assets based on
    the ratio of Expected Return / Standard Deviation (mu / sqrt(sigma_ii)).
    
    Returns:
        selection (np.array): Binary vector of the selection.
        value (float): The objective value of this selection in the QP context.
        ratio_str (str): The bitstring representation.
    """
    # 1. Handle Pandas vs Numpy types
    mu_arr = mu.values if hasattr(mu, 'values') else mu
    sigma_arr = sigma.values if hasattr(sigma, 'values') else sigma
    
    # 2. Calculate the Metric (Return / Volatility) for each asset
    # Volatility is the square root of Variance (diagonal of sigma)
    ratios = []
    for i in range(len(mu_arr)):
        variance = sigma_arr[i][i]
        volatility = np.sqrt(variance)
        
        # Avoid division by zero
        if volatility > 1e-9:
            metric = mu_arr[i] / volatility
        else:
            metric = 0.0
        ratios.append(metric)
            
    # 3. Pick the Top 'Budget' Assets
    # argsort sorts ascending, so we take the last 'budget' indices
    top_indices = np.argsort(ratios)[-budget:]
    
    # 4. Construct the Binary Selection Vector
    x = np.zeros(len(mu_arr))
    for idx in top_indices:
        x[idx] = 1.0
        
    # 5. Evaluate this selection using the SAME QP as the other solvers
    # This ensures the "Value" is directly comparable (includes risk penalties etc.)
    value = qp.objective.evaluate(x)
    
    # Create string representation (e.g. "1100")
    selection_str = "".join(str(int(i)) for i in x)
    
    return x, value, selection_str

def calculate_portfolio_energy(selection: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_factor: float) -> float:
    """
    Computes the Mean-Variance Utility (Energy) of a specific portfolio selection.
    
    Formula: Utility = (Expected Return) - (Risk Factor * Portfolio Risk)
    
    Args:
        selection (np.ndarray): A binary vector indicating which assets are selected.
        mu (np.ndarray): Expected returns vector.
        sigma (np.ndarray): Covariance matrix.
        risk_factor (float): The penalty factor for risk (q).
        
    Returns:
        float: The utility score. Higher is better. 
               (Note: This is the negative of the standard Hamiltonian energy).
    """
    # 1. Calculate Expected Return
    # Sum of returns for all selected assets (dot product).
    expected_return = np.dot(mu, selection)
    
    # 2. Calculate Portfolio Risk
    # Standard formula: x^T * Sigma * x
    # This captures the correlations between assets, even if the greedy solver ignored them.
    # We use @ for matrix multiplication.
    risk = selection @ sigma @ selection
    
    # 3. Combine into Utility Score
    energy = expected_return - (risk_factor * risk)
    
    return energy

def run_penalty_analysis(qp_base, reps, mu, sigma, q, budget, penalty_range):
    """
    Varies the penalty parameter and analyzes how it affects QAOA performance.
    
    Args:
        qp_base (QuadraticProgram): The base portfolio optimization problem.
        reps (int): Number of QAOA circuit layers (depth p).
        mu (pd.Series or np.ndarray): Expected asset returns.
        sigma (pd.DataFrame or np.ndarray): Asset covariance matrix.
        q (float): Risk factor in the portfolio problem.
        budget (int): Budget constraint (number of assets to select).
        penalty_range (list): List of penalty values to test.
    
    Returns:
        dict: Results organized by penalty value, each containing:
            - "ratio" (float): Approximation ratio for the solution.
            - "success_prob" (float): Probability of finding the exact optimal solution.
            - "optimal_str" (str): The bitstring of the best solution found.
            - "optimal_val" (float): The objective value of the solution.
    """
    results_by_penalty = {}
    
    # Find the true optimal solution once (independent of penalty)
    exact_mes = NumPyMinimumEigensolver()
    exact_opt = MinimumEigenOptimizer(exact_mes)
    result_min = exact_opt.solve(qp_base)
    true_optimal_str = "".join(str(int(x)) for x in result_min.x)
    
    print(f"True Optimal Target: {true_optimal_str}")
    print(f"{'Penalty':<10} | {'Ratio':<10} | {'Success Prob':<15} | {'Best String':<15}")
    print("-" * 60)
    
    for penalty in penalty_range:
        algorithm_globals.random_seed = 1234
        
        sampler = Sampler(backend_options={"seed_simulator": 1234})
        optimizer = SPSA(maxiter=100)
        
        qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
        qaoa = MinimumEigenOptimizer(qaoa_mes, penalty=penalty)
        result = qaoa.solve(qp_base)
        
        # Calculate Approximation Ratio
        ratio = get_aproximation_ratio(result, mu, sigma, q, budget)
        
        # Extract optimal string found
        found_optimal_str = "".join(str(int(x)) for x in result.x)
        
        # Extract Probabilities
        eigenstate = result.min_eigen_solver_result.eigenstate
        probabilities = (
            eigenstate.binary_probabilities()
            if hasattr(eigenstate, 'binary_probabilities')
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        )
        
        # Check for the true optimal solution probability
        success_prob = probabilities.get(true_optimal_str, 0.0)
        if success_prob == 0.0:
            success_prob = probabilities.get(true_optimal_str[::-1], 0.0)
        
        # Store results
        results_by_penalty[penalty] = {
            "ratio": ratio,
            "success_prob": success_prob,
            "optimal_str": found_optimal_str,
            "optimal_val": result.fval
        }
        
        print(f"{penalty:<10.1f} | {ratio:<10.4f} | {success_prob:<15.4f} | {found_optimal_str:<15}")
    
    return results_by_penalty

def run_noise_analysis_prob(qp, penalty, reps, noise_range,n_noise,budget_noise):
    """
    Runs QAOA with increasing noise and plots the PROBABILITY of finding the optimal solution.
    """
    success_probabilities = []
    
    # 1. First, find the "True" optimal string using the exact solver
    # We need this to know what to look for in the noisy results
    from qiskit_algorithms import NumPyMinimumEigensolver
    exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    # Convert [1. 0. 1. 0.] -> "1010" (No reversal needed for comparison if consistent)
    # Note: Qiskit probabilities are often Little Endian (reversed). 
    # Let's rely on the exact solver's string format.
    true_optimal_str = "".join(str(int(x)) for x in exact_result.x)
    
    print(f"True Optimal Target: {true_optimal_str}")
    print("-" * 60)
    print(f"{'Noise Prob':<15} | {'Success Prob':<15} | {'Top Found'}")

    for prob in noise_range:
        algorithm_globals.random_seed = 1234
        
        #Noise Model Setup
        noise_model = NoiseModel()
        error_1q = depolarizing_error(prob, 1)
        error_2q = depolarizing_error(prob, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        #Run QAOA

        # This forces the random number generator for the measurement shots to be identical every time.
        noisy_sampler = Sampler(
            backend_options={
                "noise_model": noise_model,
                "seed_simulator": 1234
            }
        )
        optimizer = POWELL(maxiter=100)
        qaoa_mes = QAOA(sampler=noisy_sampler, optimizer=optimizer, reps=reps)
        qaoa = MinimumEigenOptimizer(qaoa_mes, penalty=penalty)
        
        result = qaoa.solve(qp)
        
        # --- EXTRACT PROBABILITY ---
        eigenstate = result.min_eigen_solver_result.eigenstate
        probs = (
            eigenstate.binary_probabilities()
            if hasattr(eigenstate, 'binary_probabilities')
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
        )
        
        # We need to handle the bitstring reversal (Endianness)
        # Qiskit results are often right-to-left. 
        # Check both the string and its reverse to be safe, or stick to one convention.
        # Here we check if the true_optimal_str (or its reverse) is in the keys.
        
        # Try direct match first
        current_prob = probs.get(true_optimal_str, 0.0)
        
        # If 0, try checking the reverse key (common issue with Qiskit/Numpy conversion)
        if current_prob == 0.0:
            current_prob = probs.get(true_optimal_str[::-1], 0.0)
            
        success_probabilities.append(current_prob)
        
        # Find what the most probable result actually was for logging
        top_found = max(probs, key=probs.get)
        
        print(f"{prob:<15.4f} | {current_prob:<15.4f} | {top_found}")

    #Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(noise_range, success_probabilities, marker='o', color='red', linewidth=2)
    plt.title(f"QAOA Robustness: Probability of Finding Optimal 'n = {n_noise}' 'budget = {budget_noise}'", fontsize=14)
    plt.xlabel("Depolarizing Noise Probability", fontsize=12)
    plt.ylabel("Probability of Optimal Solution", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()