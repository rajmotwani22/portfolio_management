#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:17:29 2024

@author: raj
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection
assets = ['SPY', 'TLT', 'GLD']  # S&P 500, US Treasury Bond ETF, Gold ETF
data = yf.download(assets, start='2019-01-01', end='2024-01-01')['Adj Close']

# Step 2: Data Preparation
returns = data.pct_change().dropna()
normalized_returns = (1 + returns).cumprod() * 100

# Step 3: Portfolio Allocation
weights = np.array([0.5, 0.3, 0.2])
portfolio_returns = returns.dot(weights)
cumulative_returns = (1 + portfolio_returns).cumprod() * 100

# Step 4: Performance Metrics
volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
max_drawdown = cumulative_returns.div(cumulative_returns.cummax()).min() - 1

# Step 5: Visualization
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Portfolio')
plt.plot(normalized_returns, label=assets)
plt.title('Portfolio vs. Asset Performance')
plt.legend()
plt.show()

# Additional Visualization 1: Rolling Volatility
rolling_volatility = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
plt.figure(figsize=(10, 6))
plt.plot(rolling_volatility, label='Rolling Volatility (21-day)')
plt.title('Portfolio Rolling Volatility')
plt.legend()
plt.show()

# Additional Visualization 2: Correlation Heatmap
correlation_matrix = returns.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Asset Correlation Heatmap')
plt.show()

# Additional Visualization 3: Portfolio Value Distribution
plt.figure(figsize=(8, 6))
sns.histplot(portfolio_returns, kde=True, bins=30)
plt.title('Portfolio Daily Return Distribution')
plt.show()

# Step 6: Predictive Analysis - Simple Linear Regression
# Prepare the data
X = portfolio_returns[:-1].values.reshape(-1, 1)  # Previous day's returns
y = portfolio_returns[1:].values  # Next day's returns

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.6f}')

# Visualization: Actual vs Predicted Returns
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Returns')
plt.plot(y_pred, label='Predicted Returns', linestyle='--')
plt.title('Actual vs Predicted Portfolio Returns')
plt.legend()
plt.show()
