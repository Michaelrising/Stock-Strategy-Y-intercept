import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_pre_processing import *
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, pars):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, pars)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, pars)
            self.counter = 0

    def save_checkpoint(self, val_loss, pars):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(pars, self.path)
        self.val_loss_min = val_loss

# lstm model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.get_device())
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.get_device())
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 128, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):

    data_raw = np.array(stock['last'])  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    # data = np.expand_dims(data, 0)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train =  np.expand_dims(data[:train_set_size, :-1], 2)
    y_train = data[:train_set_size, -1].reshape(train_set_size, -1)

    x_test =  np.expand_dims(data[train_set_size:, :-1], 2)
    y_test = data[train_set_size:, -1].reshape(test_set_size, -1)

    return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
    summary_dir = '../Log/' + datetime.now().strftime("%Y%m%d-%H%M")
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)
    num_epochs = 5000
    # create model
    input_dim = 1
    hidden_dim = 128
    num_layers = 2
    output_dim = 1
    device ='cuda:1'
    look_back = 60  # choose sequence length
    all_stocks = StockData().all_his_of_tickers()
    model_dict = {}
    for ticker in all_stocks:
        print('=========================================')
        print('The current training stock is: ' + ticker)
        print('=========================================')
        stock_ticker = all_stocks[ticker]
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        stock_ticker['last'] = scaler1.fit_transform(stock_ticker['last'].values.reshape(-1, 1))
        stock_ticker['volume'] = scaler2.fit_transform(stock_ticker['volume'].values.reshape(-1, 1))
        x_train, y_train, x_test, y_test = load_data(stock_ticker, look_back)
        # make training and test sets in torch
        x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
        x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
        y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)
        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_test.shape = ', x_test.shape)
        print('y_test.shape = ', y_test.shape)

        stock_ticker=stock_ticker.fillna(method='ffill')

        # Train model
        #####################
        hist = np.zeros(num_epochs)

        # Number of steps to unroll
        seq_dim = look_back - 1

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        model.to(device)

        loss_fn = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=15)

        optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
        y_test = scaler1.inverse_transform(y_test.cpu().detach().numpy())
        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0 and t != 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            writer.add_scalar(ticker+'/loss', loss.item())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t > 500 and t % 5 == 0:
                # make predictions
                y_test_pred = model(x_test)
                y_test_pred = scaler1.inverse_transform(y_test_pred.cpu().detach().numpy())
                testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
                writer.add_scalar(ticker + '/test_score', testScore)
                early_stopping(testScore, model.parameters())
            if early_stopping.early_stop:
                break
        model_dict[ticker] = model.parameters()

        # invert predictions
        y_train_pred = scaler1.inverse_transform(y_train_pred.cpu().detach().numpy())
        y_train = scaler1.inverse_transform(y_train.cpu().detach().numpy())


        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0],y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0],y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # Visualising the results
        plt.style.use('seaborn')
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        axes.plot(stock_ticker[len(stock_ticker) - len(y_test):]['date'], y_test, color='red', label='Real' + ticker +' Stock Price')
        axes.plot(stock_ticker[len(stock_ticker) - len(y_test):]['date'], y_test_pred, color='blue',
                  label='Predicted ' + ticker +' Stock Price')
        plt.title(ticker + ' Stock Price Prediction')
        plt.xlabel('Time')
        # ticks = stock_ticker[len(stock_ticker) - len(y_test):]['date']
        # tick_pos = [i if ticks.shape[0]/10 == 0 else None for i in range(ticks.shape[0])]
        plt.xticks([])
        plt.ylabel(ticker + ' Stock Price')
        plt.legend()
        plt.savefig(summary_dir + '/' + ticker +'_pred.png')
        # plt.show()

