import math
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from torch.nn import Module, LSTM, Linear
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

import logging,sys
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd

class Config:

    feature_columns = list(range(3, 11 + 1))       # 8 characteristics
    label_columns = [11]

    # feature_columns = list(range(3, 12 + 1))     # 9 characteristics
    # label_columns = [12]

    # feature_columns = list(range(3, 17 + 1))     # all characteristics
    # label_columns = [17]

    # feature_columns = list(range(3, 10 + 1))     # ETF500
    # label_columns = [10]

    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)
    predict_day = 1             # Forecast the next few days

    input_size = len(feature_columns)-1
    output_size = len(label_columns)

    hidden_size = 128           # LSTM hidden layer size, also output size
    lstm_layers = 2             # Stacking layers of LSTM
    dropout_rate = 0.2          # dropout rate
    time_step = 5               # window size

    do_train = True
    do_predict = True
    add_train = False
    shuffle_train_data = True
    use_cuda = False

    train_data_rate = 0.95      # The proportion of training data to total data
    valid_data_rate = 0.15      # Ratio of validation data to training data

    batch_size = 64
    learning_rate = 1e-3        # Learning rate
    epoch = 500
    patience = 15               # Train the epoch, the verification set will stop before it is promoted
    random_seed = 42            # Random seed

    do_continue_train = False
    continue_flag = ""
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    debug_mode = False
    debug_num = 500

    used_frame = "pytorch"
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # Path parameter
    file_save_path = ""
    train_data_path = ""
    model_save_path = ""
    figure_save_path = ""
    log_save_path = ""
    record_save_path = ""
    logger_name = ""

    do_log_print_to_screen = True
    do_log_save_to_file = True           # Whether to log config and training procedures
    do_figure_save = True
    do_train_visualized = False

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.mean = np.mean(self.data, axis=0)              # The mean and variance of the data
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # Normalization
        self.start_num_in_test = 0

    def read_data(self):
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        init_data = init_data.fillna(method='bfill')        # backfill
        return init_data.values, init_data.columns.tolist()

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num, 0:self.norm_data.shape[1]-1]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]

        if not self.config.do_continue_train:
            # In the non-continuous training mode, each time_step row will be taken as a sample, and the two samples will be staggered by one line
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # In continuous training mode, the data of each time_step line will be taken as a sample, and the two samples will stagger the time_step line.
            # In this way, the final_state of the previous sample can be used as the init_state of the next sample, and cannot be shuffled
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
        train_x, train_y = np.array(train_x), np.array(train_y)
        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return self.train_x, self.valid_x, self.train_y, self.valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:,0:self.norm_data.shape[1]-1]
        sample_interval = min(feature_data.shape[0], self.config.time_step)
        self.start_num_in_test = feature_data.shape[0] % sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # In the test data, each time_step row is taken as a sample, and the two samples stagger the time_step rows
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)
    return logger


'''pytorch prediction model includes LSTM timing prediction layer and Linear regression output layer'''
class Net(Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden

'''Model training'''
def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU training or GPU
    model = Net(config).to(device)      # For GPU training,.to(device) will copy the model/data to GPU video memory
    if config.add_train:                # If the training is incremental, the original model parameters will be loaded first
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0

    train_loss_all = []  # Record the loss of training for each epoch of the training set
    vaild_loss_all = []  # Record the loss of each epoch validation in the validation set

    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()
            pred_Y, hidden_train = model(_train_X, hidden_train)

            if not config.do_continue_train:
                hidden_train = None             # If the training is not continuous, reset the hidden
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)
            loss = criterion(pred_Y, _train_Y)  # Calculated loss
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:   # It is displayed every hundred steps
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # The following is the early stop mechanism. When the model training has not improved the prediction effect of the verification set for successive periods of patience
        # It will stop to prevent overfitting
        model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            if not config.do_continue_train: hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        train_loss_all.append(train_loss_cur)
        vaild_loss_all.append(valid_loss_cur)

        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))
        if config.do_train_visualized:
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # Save model
            return model
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # If the validation set index does not improve for consecutive patience epochs, the training will be stopped
                logger.info(" The training stops early in epoch {}".format(epoch))
                return model
                break

    draw_loss(config=config, train_loss_all=train_loss_all, valid_loss_all=vaild_loss_all) # Charting loss


'''Model testing'''
def predict(config, test_X):
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # Load model parameter

    result = torch.Tensor().to(device)

    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()

'''Restore data'''
def restore_data(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # Restore the data by saving the mean and variance
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    return label_data, predict_data

'''Forecast result graph'''
def draw(config: Config, origin_data: Data, stock_record):
    tradeDate = stock_record.tradeDate
    true_data = stock_record['true_target']
    predict_data = stock_record['predict_target']

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    x_major_locator = MultipleLocator(15)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=15)
    plt.plot(tradeDate, true_data, label='true')
    plt.plot(tradeDate, predict_data, label='predict')
    plt.grid()
    plt.legend(prop={'size': 20})
    plt.title("Predict stock {} price with {}".format(label_name, config.used_frame))
    if config.do_figure_save:
        plt.savefig(config.figure_save_path + "{}predict_{}_with_{}.png".format(config.continue_flag, label_name, config.used_frame))
    plt.show()

'''Loss graph'''
def draw_loss(config: Config, train_loss_all: [], valid_loss_all: []):
    plt.plot(train_loss_all, label='train_loss')
    plt.plot(valid_loss_all, label='valid_loss')
    plt.grid()
    plt.legend()
    plt.title("train_loss and valid_loss")
    if config.do_figure_save:
        plt.savefig(config.figure_save_path + "train_loss and valid_loss.png")
    plt.show()

'''Record result'''
def record_result(origin_data, label_data,predict_data):
    tradeDate = origin_data.loc[(len(origin_data) - len(label_data)):,'tradeDate']
    stock_record = pd.DataFrame(columns=['tradeDate','true_target','predict_target'])
    stock_record['tradeDate'] = tradeDate
    stock_record['true_target'] = label_data[:, 0]
    stock_record['predict_target'] = predict_data[:, 0]
    return stock_record

'''Evaluation index function'''
def Evaluation_Prediction(df):
    true_target = df.true_target
    predict_target = df.predict_target

    # MSE=((y_true-y_pre)^2)/n
    et = true_target - predict_target
    MSE = ((et) ** 2).mean()

    # RMSE=sqrt(((y_true-y_pre)^2)/n)
    RMSE = math.sqrt(MSE)

    # MAE=mean(absolute(y_true – y_pre))
    MAE = abs(et).mean()

    # MAPE
    MAPE = (abs(et) / true_target).mean()

    # MDA
    sign_result = np.sign(predict_target[1:] - true_target.shift(1)[1:]) * np.sign(true_target[1:] - true_target.shift(1)[1:])
    sign_result = sign_result.replace(-1,0)
    MDA = sign_result.mean()

    # R^2
    y_mean = true_target.mean()
    R_sqrt = sum((true_target - predict_target)**2) / sum((true_target - y_mean)**2)

    # score
    score = 0.7*MDA - 0.2*MAPE - 0.1*MAE
    return score