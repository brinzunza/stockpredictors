import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def retrieve_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start = start_date, end = end_date)
    return data

stock_data = retrieve_stock_data("QQQ", "2020-01-01", "2023-01-01")

stock_data.head()

plt.figure(figsize=(12,6))
plt.plot(stock_data["Close"])
plt.title("Closing Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()

train_size = int(len(stock_data) * 0.8)
train_data, test_data = stock_data[:train_size], stock_data[train_size:]

print(len(train_data))
print(len(test_data))

train_data = train_data["Close"]
test_data = test_data["Close"]

def find_best_arima_order(data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, order=(p,d,q))
                    model_fit = model.fit()
                    mse = mean_squared_error(data, model_fit.fittedvalues)
                    if mse < best_score:
                        best_score, best_cfg = mse, (p,d,q)

                except:
                    continue

    return best_cfg

p_values = range(0,3)
d_values = range(0,3)
q_values = range(0,3)

best_cfg = find_best_arima_order(train_data, p_values, d_values, q_values)
print("Best ARIMA parameters: ARIMA{}".format(best_cfg))

model = ARIMA(train_data, order = best_cfg)
model_fit = model.fit()

forecasted_values = model_fit.forecast(steps = len(test_data))

mse = mean_squared_error(test_data, forecasted_values)
print("Mean Squared Error (MSE): {:.2f}".format(mse))

forecasted_values = model_fit.forecast(steps = 180)

forecast_dates = pd.date_range(start = test_data.index[-1], periods = 180, freq = "D")

forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted": forecasted_values})

forecast_df.head()

plt.figure(figsize= (12,6))
plt.plot(test_data.index, test_data, label = "Actual", color = "b")
plt.plot(forecast_df["Date"], forecast_df["Forecasted"], label = "Forecasted", color = "r")
plt.title("Forecasted Closing Prices for the Next 6 Months")
plt.ylabel("Closing Price")
plt.xlabel("Date")
plt.legend()
plt.show()