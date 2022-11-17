import copy
from abc import ABC
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import bernoulli
from collections import deque
from gym.utils import seeding
import torch
import argparse
from data_pre_processing import *
from lstm_model import LSTM
from sklearn.preprocessing import MinMaxScaler


class StockStrategy(gym.Env, StockData):
    def __init__(self, tickers, model_path, look_back = 60, input_dim = 1, hidden_dim = 128, num_layers = 2, output_dim = 1, device ='cuda:1'):
        super().__init__(tickers, model_path)
        self.dict_of_tickers = self.selected_tickers(tickers)
        # tickers: denotes the strategy of buying or selling of these stocks
        self.SelectTickers = sorted(tickers)
        self.action_space =  spaces.Discrete(3 ** len(tickers)) # each stock has 3 strategy buy sell or hold
        self.action_sets =  tuple(np.ndindex(tuple([3 for _ in range(len(tickers))])))
        # the observation space contains all the stocks
        self.observation_space = spaces.Box(low = np.zeros(len(tickers)), high = np.ones(len(tickers)))
        self.model_dicts = {}
        self.scalers = {}
        self.start_date =  0
        for ticker in self.SelectTickers:
            scaler = MinMaxScaler(feature_range=(0, 1))
            stock_ticker = self.dict_of_tickers[ticker]
            stock_ticker['last'] = scaler.fit_transform(stock_ticker['last'].values.reshape(-1, 1))
            self.scalers[ticker] = scaler
            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
            model.load_state_dict(model_path + '/' + ticker +'.pth')
            model.eval()
            self.model_dicts[ticker] = model
            self.start_date = self.dict_of_tickers[ticker][-1] if self.start_date == 0 else max(self.start_date, self.dict_of_tickers[ticker][-1])
        self.look_back_data = {}
        self.data_pre(look_back, device)

        self.steps = 0

    def data_pre(self, look_back, device):

        for ticker in self.SelectTickers:
            stock = self.dict_of_tickers[ticker]
            model = self.model_dicts[ticker]
            last_date = stock['date'][-1]
            data_raw = np.array(stock['last'])  # convert to numpy array
            # create the last possible sequences of length look_back
            data = data_raw[len(data_raw)-look_back: ]
            data = torch.from_numpy(data.reshape(1, -1, 1)).type(torch.Tensor).to(device)
            if last_date < self.start_date:
                for i in range(self.start_date - last_date):
                    new_data = model(data)
                    new_data = new_data.cpu().detach().numpy()
                    data = torch.cat((data, new_data.view(1, -1, 1)), 1)
                    # data = np.append(data.detach().numpy().reshape(-1), new_data)
                    data = data[:, 1:, :] # torch.from_numpy(data[1:].reshape(1, -1, 1)).type(torch.Tensor).to(device)
            self.look_back_data[ticker] = data


    def StockStates(self):
        for ticker in self.SelectTickers:
            model = self.model_dicts[ticker]






    def step(self, action):
        # if self.steps == 0:

        # the state space is embedded as the whole stock markets
        actions = self.action_sets[action]




















