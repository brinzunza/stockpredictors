# Stock Price Forecasting using ARIMA

Using autoregressive integrated moving average (ARIMA) to forecast stock prices. The stock data is retrieved using the yfinance library, and ARIMA models are used to make predictions based on historical data. We evaluate the model’s performance using Mean Squared Error (MSE) and visualizes both the actual and forecasted stock prices.

## Features

	•	Retrieve stock price data from Yahoo Finance using yfinance.
	•	Visualize historical closing prices over a specified date range.
	•	Split the dataset into training and testing sets.
	•	Use grid search to find the best ARIMA model parameters (p, d, q).
	•	Forecast future stock prices. 
	•	Visualize the forecasted prices and compare them to the actual prices.
	•	Calculate the Mean Squared Error (MSE) to evaluate model performance.


## Set-up

pip install -requirements.txt