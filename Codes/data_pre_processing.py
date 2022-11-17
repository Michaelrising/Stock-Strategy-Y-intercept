import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


class StockData:
    def __init__(self, file_path ='../Data/data.csv'):
        self.all_data = pd.read_csv(file_path, header=0)
        self.tickers = set(list(self.all_data['ticker']))
        self.dict_of_tickers = {}

    def his_of_ticker(self, ticker):
        if ticker in self.tickers:
            ticker_his = self.all_data[self.all_data['ticker']==ticker]
        else:
            raise ValueError
        return ticker_his

    def all_his_of_tickers(self):
        for ticker in self.tickers:
            self.dict_of_tickers[ticker] = self.his_of_ticker(ticker)
        return self.dict_of_tickers

    def plot_ts_of_tickers(self, tickers, plot_style='seaborn'):
        plot_tickers = []
        for ticker in tickers:
            if ticker in self.tickers:
                plot_tickers.append(ticker)

        ticker_num = len(plot_tickers)
        # if ticker_num % 2 == 0:
        plt.style.use(plot_style)
        fig = plt.figure()
        for i in range(ticker_num):
            plt.subplot(math.ceil(ticker_num/2), 2, i+1)
            plot_data = self.dict_of_tickers[tickers[i]]
            # plot_data_price = self.all_his_of_tickers[tickers[i]].iloc[1]
            # plot_data_volume = self.all_his_of_tickers[tickers[i]].iloc[2]
            plt.plot(plot_data['last'])
            # plt.plot(plot_data['volume'])
        plt.show()




file_path = '../Data/data.csv'
stocks = StockData(file_path)
all_his_of_tickers = stocks.all_his_of_tickers()













