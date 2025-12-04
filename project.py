import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#Test
tickers = ["AAPL", "GOOG", "MSFT", "AMZN"]
start_date = "2023-01-01"
end_date = "2024-01-01"


mu_raw, sigma_raw, prices = get_portfolio_data(tickers, start_date, end_date)

#Normalization
mu_norm, sigma_norm = normalize_data(mu_raw, sigma_raw)

#print("Normalized Expected Returns (mu):\n", mu_norm)
#print("\nNormalized Covariance (sigma):\n", sigma_norm)
#print("Prices: ", prices)

#Plot the covariance matrix
plt.imshow(sigma_norm, interpolation='nearest')
plt.colorbar()
plt.title("Normalized Covariance Matrix")
plt.show()