import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_pre_processing import *
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Here we define our model
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 128, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):

    data_raw = stock['last']  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
    num_epochs = 1000
    # create model
    input_dim = 2
    hidden_dim = 128
    num_layers = 2
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    # print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    look_back = 60  # choose sequence length
    all_stocks = StockData().all_his_of_tickers()
    for ticker in all_stocks:
        stock_ticker = all_stocks[ticker]
        x_train, y_train, x_test, y_test = load_data(stock_ticker, look_back)
        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_test.shape = ', x_test.shape)
        print('y_test.shape = ', y_test.shape)

        stock_ticker=stock_ticker.fillna(method='ffill')

        scaler = MinMaxScaler(feature_range=(-1, 1))
        stock_ticker['last'] = scaler.fit_transform(stock_ticker['last'].values.reshape(-1,1))

        # Train model
        #####################
        hist = np.zeros(num_epochs)

        # Number of steps to unroll
        seq_dim = look_back - 1

        for t in range(num_epochs):
            y_train_pred = model(x_train)

            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0 and t != 0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # make predictions
        y_test_pred = model(x_test)

        # invert predictions
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
        y_train = scaler.inverse_transform(y_train.detach().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = scaler.inverse_transform(y_test.detach().numpy())

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()

        axes.plot(stock_ticker[len(stock_ticker) - len(y_test):].index, y_test, color='red', label='Real' + ticker +' Stock Price')
        axes.plot(stock_ticker[len(stock_ticker) - len(y_test):].index, y_test_pred, color='blue',
                  label='Predicted ' + ticker +' Stock Price')
        plt.title(ticker + ' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(ticker + ' Stock Price')
        plt.legend()
        plt.savefig(ticker+'_pred.png')
        plt.show()

