
import numpy as np

class GreedyAlgorithm:
    """
    Performs a greedy algorithm that sorts by μ/∑ and picks the top k.
    """
    
    def run(self, mu: np.ndarray, covariance: np.ndarray, budget: int) -> np.ndarray:
        """Selects the top k components of the vector."""
        sqrt_cov = np.sqrt(np.diag(covariance))
        selected = np.argsort(sqrt_cov)[-budget:]  # Selects the top indices
        solution = np.zeros_like(mu, dtype=int)
        solution[selected] = 1
        return solution

    def get_energy(self, mu: np.ndarray, covariance: np.ndarray, risk_factor: float, budget: int) -> float:
        """Computes the portfolio objective C(z) = μ^T z - λ z^T Σ z."""
        z = self.run(mu, covariance, budget)
        expected_return = np.dot(mu, z)
        risk = np.dot(z, covariance @ z)  # z^T Σ z
        energy = expected_return - risk_factor * risk
        return energy
