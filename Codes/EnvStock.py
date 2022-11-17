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
from lstc_model import LSTM


class StockStrategy(gym.Env, StockData):
    def __init__(self, tickers, model_path, input_dim = 1, hidden_dim = 128, num_layers = 2, output_dim = 1, device ='cuda:1'):
        super().__init__(tickers, model_path)
        self.dict_of_tickers = self.all_his_of_tickers()
        # tickers: denotes the strategy of buying or selling of these stocks
        self.SelectTickers = tickers
        self.action_space =  spaces.Discrete(3 ** len(tickers)) # each stock has 3 strategy buy sell or hold
        self.action_sets =  tuple(np.ndindex(tuple([3 for _ in range(len(tickers))])))
        # the observation space contains all the stocks
        self.observation_space = spaces.Box(low = np.zeros(len(self.tickers)), high = np.ones(len(self.tickers)))
        self.model_dicts = {}
        self.tickers = list(self.tickers)
        self.tickers.sort()
        for ticker in self.tickers:
            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
            model.eval()
            self.model_dicts[ticker] = model

    def step(self, action):
        actions = self.action_sets[action]













