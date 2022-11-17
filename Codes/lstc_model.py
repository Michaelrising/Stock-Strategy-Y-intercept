import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_pre_processing import *
from torch.utils.tensorboard import SummaryWriter

class lstm_model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(lstm_model, self).__init__()
        # lstm takes the normalized stock values and volumes as input
        self.linear1 =  nn.Linear(input_dim, hidden_dim)
        # Building LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first = True)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, stock):
        # Initialize hidden and cell state with zero
        h0 = torch.zeros(self.num_layers, stock.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, stock.size(0), self.hidden_dim).requires_grad_().cuda()
        lstm_out, _, _= self.lstm(stock.view(stock.shape[0][0], 1, -1), (h0.detach(), c0.detach()))
        pre_price = self.linear(lstm_out.view(stock.shape[0][0], -1))

        return pre_price


stocks = StockData(file_path)

def train(stocks, ticker):
    ticker_stock = stocks.his_of_ticker(ticker)
    stock_lenth = len(ticker_stock)
    train_len = int(0.8*stock_lenth)

    # LSTM model
    input_dim = 6
    hidden_dim = 512
    num_layers = 6
    output_dim = 1
    model = lstm_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    # k = torch.rand(8, 20, 2).cuda()  # random input
    # with SummaryWriter(comment='lstm') as w:
    #     w.add_graph(model, (k,))
    # print(model)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)







