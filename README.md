Portfolio Management: Analysis and Prediction
Introduction
This project analyzes the historical performance of a diversified investment portfolio consisting of stocks, bonds, and gold. The primary objectives are to assess the portfolio's risk and return characteristics, visualize key performance metrics, and develop a basic predictive model to forecast future returns.
Data Collection
The data was sourced from Yahoo Finance, covering the daily closing prices of the S&P 500 (SPY), a US Treasury Bond ETF (TLT), and a Gold ETF (GLD) over the past five years.
Portfolio Allocation
The portfolio allocation was set as 50% stocks (SPY), 30% bonds (TLT), and 20% gold (GLD). The weighted returns were calculated based on these allocations to track the portfolio's performance.
Performance Metrics
Key performance metrics were calculated:
•	Annualized Volatility: Measures the risk of the portfolio.
•	Sharpe Ratio: Assesses the risk-adjusted return.
•	Maximum Drawdown: Identifies the largest peak-to-trough decline in the portfolio's value.
Visualizations
Several visualizations were created to better understand the portfolio:
•	Portfolio vs. Asset Performance: Compares the cumulative returns of the portfolio against individual assets.
 

•	Rolling Volatility: Displays the portfolio's volatility over time.
 

•	Correlation Heatmap: Illustrates the relationships between the assets.
 

•	Portfolio Daily Return Distribution: Shows the distribution of daily returns.
 


Predictive Analysis
A simple linear regression model was used to predict the next day's portfolio return based on past returns. The model's performance was evaluated using Mean Squared Error (MSE).
Conclusion
This project showcases the application of Python in portfolio management, providing insights into the portfolio's risk-return profile and laying the groundwork for future enhancements in predictive modeling and portfolio optimization.

