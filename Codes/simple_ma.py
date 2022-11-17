
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_pre_processing import *
pd.options.mode.chained_assignment = None

def simple_ma(stocks, ticker):
    ticker_stock = stocks[ticker]
    # moving average to find the trend of this stock

    ticker_stock['MA15'] = ticker_stock['last'].rolling(15).mean()
    ticker_stock['MA50'] = ticker_stock['last'].rolling(50).mean()
    ticker_stock = ticker_stock.dropna()

    # strategy for this simple MA is As the moving average of the
    # stock price is higher in the last 15 days, it shows a positive trend.

    conditions = [ticker_stock['MA15'] > ticker_stock['MA50'],
                  ticker_stock['MA15'] < ticker_stock['MA50']]

    choices = [1, 0]

    ticker_stock['Position'] = np.select(conditions, choices)

    # First we calculate the daily profit.
    ticker_stock['last_Next'] = ticker_stock['last'].shift(-1)
    ticker_stock['Profit'] = [ticker_stock.loc[i, 'last_Next'] - ticker_stock.loc[i, 'last'] if ticker_stock.loc[i, 'Position'] == 1 else 0 for i in ticker_stock.index]

    # Then the cumulative profit with cumsum() method
    ticker_stock['Cum_Profit'] = ticker_stock['Profit'].cumsum()

    # Plot it
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(ticker_stock['Cum_Profit'], color='#006BA2', linewidth=3)
    plt.xlabel('Date')
    plt.title(ticker + ' Stock Price Cumulative Profit')
    plt.show()

    return ticker_stock['Cum_Profit'][-1]


def exponential_ma(stocks, ticker):
    ticker_stock = stocks[ticker]
    stock_lenth = len(ticker_stock)
    train_len = int(0.8 * stock_lenth)
    train_ticker = ticker_stock[:train_len]
    test_ticker = ticker_stock[train_len:]

    window_size = 100

    run_avg_predictions = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5
# calculate the exponential moving average of stock
    for pred_idx in range(1, train_len):
        running_mean = running_mean * decay + (1.0 - decay) * train_ticker['last'][pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_ticker['last'][pred_idx]) ** 2)
        # run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

    plt.figure()
    plt.plot(range(ticker_stock.shape[0]), ticker_stock['last'], color='b', label='True')
    plt.plot(range(0, train_len), run_avg_predictions, color='orange', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()


if __name__ == '__main__':
    file_path = '../Data/data.csv'
    stocks = StockData(file_path)
    all_stocks_his = stocks.all_his_of_tickers()
    all_profit = {}
    for ticker in stocks.tickers:
        profit = simple_ma(all_stocks_his, ticker)
        all_profit[ticker] = profit

    mean_profit = np.mean(all_profit)
    buy = []
    for ticker in all_profit:
        if all_profit[ticker] >= mean_profit:
            buy.append(ticker)



