import numpy as np

def normalize_data(mu: np.ndarray, sigma: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Normalizes data to [0, 1] range."""
    mu_min = mu.min()
    mu_max = mu.max()
    mu_normalized = (mu - mu_min) / (mu_max - mu_min) #Normalization between [0,1]

    sigma_max = sigma.max().max()
    sigma_min = sigma.min().min()
    sigma_normalized = (sigma-sigma_min) / (sigma_max-sigma_min) #Normalization of sigma (the covariance matrix)
    
    return mu_normalized, sigma_normalized