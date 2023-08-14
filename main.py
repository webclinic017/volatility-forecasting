
# Retrieving Historical Price Data from `yfinance`

import yfinance as yf

# Retrieve historical volatility data for GS
stock = yf.Ticker("GS")
volatility_data = stock.history(period="max")

# Exploratory Data Analysis (EDA) of Price Data

import matplotlib.pyplot as plt

# Plot the historical volatility data
plt.figure(figsize=(10, 6))
plt.plot(volatility_data.index, volatility_data["Close"])
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Historical Volatility Data")
plt.grid(True)

plt.show()

# Calculate the rolling mean and standard deviation

rolling_mean = volatility_data["Close"].rolling(window=30).mean()
rolling_std = volatility_data["Close"].rolling(window=30).std()

# Plot the rolling mean and standard deviation

plt.figure(figsize=(10, 6))
plt.plot(volatility_data.index, volatility_data["Close"], label="Volatility")
plt.plot(rolling_mean.index, rolling_mean, label="Rolling Mean")
plt.plot(rolling_std.index, rolling_std, label="Rolling Std")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Rolling Mean and Standard Deviation of Volatility Data")
plt.legend()
plt.grid(True)

plt.show()

# Implement the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) Model

import numpy as np
from arch import arch_model

# Calculate log returns
# Log returns represent the percentage change in the volatility data from one period to the next.
# Commonly used in financial analysis as they provide a more meaningful representation of the data.
returns = np.log(volatility_data['Close']).diff().dropna()

# Fit the GARCH(1,1) model
# includes one lag of both the returns and the conditional variance.
model = arch_model(returns, vol="Garch", p=1, q=1)
results = model.fit()

# Estimating and Forecasting Volatility
# The estimated volatility represents the conditional variance of the log returns, while the forecasted volatility provides future volatility predictions.

# Estimate the volatility
volatility = results.conditional_volatility

# Plot the estimated and actual volatility
plt.figure(figsize=(10, 6))
plt.plot(volatility.index, volatility, label="Estimated Volatility")
plt.plot(returns.index, returns, label="Actual Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Estimated and Actual Volatility")
plt.legend()
plt.grid(True)

plt.show()

'''
The plot shows the estimated volatility (conditional variance) and the actual volatility (log returns) over time. It helps us assess the accuracy of the volatility estimates.
'''

# Forecast the future volatility using the fitted model

# Forecast the volatility
forecast = results.forecast(start=0, horizon=30)
forecast_volatility = forecast.variance.dropna().values.flatten()

# Plot the forecasted volatility
plt.figure(figsize=(10, 6))
plt.plot(forecast_volatility, label="Forecasted Volatility")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.title("Forecasted Volatility")
plt.legend()
plt.grid(True)

plt.show()

# Evaluating the Model Performance

# Calculate the mean absolute error (MAE)
mae = np.mean(np.abs(volatility - returns))
print("Mean Absolute Error (MAE):", mae)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((volatility - returns) ** 2))
print("Root Mean Squared Error (RMSE):", rmse)

"""
The MAE and RMSE provide measures of the average and overall forecast errors, respectively. Lower values indicate better model performance.
"""

# Calculate the forecast errors
errors = volatility - returns

# Plot the histogram of forecast errors
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, density=True)
plt.xlabel("Forecast Error")
plt.ylabel("Density")
plt.title("Histogram of Forecast Errors")
plt.grid(True)

plt.show()

"""
Conclusion

In this tutorial, we developed a volatility forecasting model using Python. We retrieved historical volatility data using the yfinance library, performed exploratory data analysis (EDA), implemented the GARCH model, estimated and forecasted volatility, and evaluated the model performance.

Volatility forecasting plays a crucial role in financial analysis and risk management. By accurately predicting future volatility, traders and investors can make informed decisions and effectively manage their portfolios.
Remember to experiment with different model specifications and data sources to improve the accuracy of your volatility forecasts. Volatility forecasting is a complex task that requires continuous learning and refinement.

"""


