import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from math import floor
from data_pre_processing import *


# initial data processing
file_path = '../Data/data.csv'
stocks = StockData(file_path)


# STOCHASTIC OSCILLATOR CALCULATION
