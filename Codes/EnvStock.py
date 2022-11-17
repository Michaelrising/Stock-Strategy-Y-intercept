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
    def __init__(self, tickers, model_path, money, pred_T, r, look_back=60, input_dim=1, hidden_dim=128, num_layers=2,
                 output_dim=1, device='cuda:1'):
        # money is the initial money u have, r is the interest rate of risk-free investment
        super().__init__()
        self.dict_of_tickers, _, end_date = self.selected_tickers(tickers)
        # tickers: denotes the strategy of buying or selling of these stocks
        self.SelectTickers = sorted(tickers)
        self.action_space =  spaces.Discrete(3 ** len(tickers)) # each stock has 3 strategy buy sell or hold
        self.action_sets =  tuple(np.ndindex(tuple([3 for _ in range(len(tickers))])))
        # the observation space contains all the stocks
        self.observation_space = spaces.Box(low = np.zeros(len(tickers)), high = np.ones(len(tickers)))
        self.model_dicts = {}
        self.scalers = {}
        self.start_date =  max(end_date)
        for ticker in self.SelectTickers:
            scaler = MinMaxScaler(feature_range=(0, 1))
            stock_ticker = self.dict_of_tickers[ticker]
            # transform to range [0, 1]
            stock_ticker['last'] = scaler.fit_transform(stock_ticker['last'].values.reshape(-1, 1))
            self.scalers[ticker] = scaler
            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
            model.load_state_dict(torch.load(model_path + '/' + ticker +'.pth'))
            model.eval()
            self.model_dicts[ticker] = model
        self.look_back_data = {}
        self.curr_stock_states = np.zeros(len(tickers))
        self.data_pre(look_back, device)
        self.steps = 0
        self.money = money
        self.pred_T = pred_T
        self.look_back = look_back
        self.device = device

    def data_pre(self, look_back, device):
        # all the tickers have to start in the same date and we force them to by prediction of pre-trained LSTM model
        for j, ticker in enumerate(self.SelectTickers):
            stock = self.dict_of_tickers[ticker]
            model = self.model_dicts[ticker]
            scaler = self.scalers[ticker]
            last_date = stock.date[stock.index[-1]]

            data_raw = np.array(stock['last'])  # convert to numpy array
            # create the last possible sequences of length look_back
            data = data_raw[len(data_raw)-look_back+1: ]
            data = torch.from_numpy(data.reshape(1, -1, 1)).type(torch.Tensor).to(device)
            if last_date < self.start_date:
                days = self.start_date - last_date
                for i in range(days.days): # days.days
                    with torch.no_grad():
                        new_data = model(data)
                    # new_data = new_data.cpu().detach().numpy()
                    data = torch.cat((data, new_data.unsqueeze(1)), 1)
                    # data = np.append(data.detach().numpy().reshape(-1), new_data)
                    data = data[:, 1:, :]
            self.curr_stock_states[j] = scaler.inverse_transform(data[:, -1, :].cpu().detach().numpy())
            self.look_back_data[ticker] = data


    def StockStates(self):
        # generate the states of given tickers
        stockstates = []
        for ticker in self.SelectTickers:
            model = self.model_dicts[ticker]
            look_back_data = self.look_back_data[ticker]
            scaler = self.scalers[ticker]
            with torch.no_grad():
                stock_state = model(look_back_data)
            look_back_data = torch.cat((look_back_data, stock_state.unsqueeze(1)), 1)

            # back to real price
            stockstates.append(scaler.inverse_transform(stock_state.cpu().detach().numpy()))
            self.look_back_data[ticker] = look_back_data[:, 1: , :]
        self.curr_stock_states = np.array(stockstates).reshape(-1)


    def step(self, action):
        # hypothesis: at first we have one share for all stocks
        actions = np.array(list(self.action_sets[action])) - 1
        states_be4_action = self.curr_stock_states
        self.StockStates()
        states_aft_action = self.curr_stock_states
        # define reward function
        prices_diff = states_aft_action - states_be4_action
        reward = 0
        # action meaning: 0: sell 1: buy 2: hold
        reward += sum(actions * prices_diff)
        self.money -= sum(actions * states_be4_action)
        done = bool(self.money<0 and self.steps <= self.pred_T)

        return states_aft_action, reward, done


    def reset(self):
        self.data_pre(self.look_back, self.device)
        self.money=10**5
        return self.curr_stock_states




























