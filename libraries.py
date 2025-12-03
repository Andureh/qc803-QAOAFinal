import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Pick your assets (The "Universe")
# Let's pick 4 distinct tech stocks for a simple proof-of-concept
tickers = ["AAPL", "GOOG", "MSFT", "NVDA"]

print(f"Downloading data for: {tickers}...")

# 2. Download historical data (e.g., from 2020 to 2023)
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Close']

# 3. Calculate the two key Physics inputs
# Log-returns are standard in finance for stability
mu = data.pct_change().mean().to_numpy() # The "Magnetic Field" (Return)
sigma = data.pct_change().cov().to_numpy() # The "Coupling Matrix" (Risk)

print("\n--- Data Ready for QAOA ---")
print(f"Expected Returns (Vector mu):\n{mu}")
print(f"Covariance Matrix (Matrix Sigma):\n{sigma}")

# Optional: Visualize what you downloaded
# (Physicists love plotting the data first)
(data / data.iloc[0] * 100).plot(figsize=(10, 6))
plt.ylabel("Normalized Price (Start = 100)")
plt.title("Asset Performance History")
plt.show()