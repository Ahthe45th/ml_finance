import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2)

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

class classifier(nn.Module):
    def __init__(self):
        '''
        Initialises the rnn model and then implicitly initialises
        everything we need to do for the text processing parts
        '''
        super(classifier, self).__init__()
        self.hidden_size = 32
        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True
            )
        self.batchnorm = nn.BatchNorm1d(self.hidden_size*2)
        self.drop_en = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(self.hidden_size*2, 32)
        self.output = nn.Linear(32, 1)
        self.intent_output = nn.Linear(32, self.numintents)
        self.still_debug = True
        # self.hidden = self.init_hidden()

    def forward(self, input1, debug=False):
        x_inmodel = self.embedding(input1.long())
        x_inmodel = self.drop_en(x_inmodel)

        out_rnn, hidden = self.lstm1(x_inmodel, None)
        row_indices = torch.arange(0, input1.size(0)).long()
        last_tensor = out_rnn[row_indices, :, :]
        last_tensor = torch.mean(last_tensor, dim=1)

        x_inmodel = self.batchnorm(last_tensor)
        x_inmodel = self.lin2(last_tensor)

        context = self.context_output(x_inmodel.float())
        intent = self.intent_output(x_inmodel.float())

        return context, intent

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

def get_tensorflow_model():
    ## define basic model class
    model = keras.models.Sequential()
    ## add layering
    model.add(keras.layers.LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=96, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=96, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=96))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_torch_model():
    ## define basic model class and layering
    model = classifier()
    return model

def save_torch_model(model, dev_model=False):
    if dev_model:
        today_str = datetime.date.today().strftime('%d_%m_%Y')
        model_location = Location.Home(__file__) + '/main_storage/development_models/stock_prediction_' + today_str + '.h5'
        torch.save(model.state_dict(), model_location)
    else:
        model_location = Location.Home(__file__) + '/main_storage/production_models/stock_prediction.h5'
        torch.save(model.state_dict(), model_location)

def save_tensorflow_model(model, dev_model=False):
    if dev_model:
        today_str = datetime.date.today().strftime('%d_%m_%Y')
        model.save(Location.Home(__file__) + '/main_storage/development_models/stock_prediction_' + today_str + '.h5')
    else:
        model.save(Location.Home(__file__) + '/main_storage/production_models/stock_prediction.h5')

def load_torch_model(model, dev_model=False):
    if dev_model:
        today_str = datetime.date.today().strftime('%d_%m_%Y')
        model_location = Location.Home(__file__) + '/main_storage/development_models/stock_prediction_' + today_str + '.h5'
    else:
        model_location = Location.Home(__file__) + '/main_storage/production_models/stock_prediction.h5'
    torchload = torch.load(model_location)
    model.load_state_dict(torchload)
    return model

def load_tensorflow_model(dev_model=False):
    if dev_model:
        today_str = datetime.date.today().strftime('%d_%m_%Y')
        model_location = Location.Home(__file__) + '/main_storage/development_models/stock_prediction_' + today_str + '.h5'
    else:
        model_location = Location.Home(__file__) + '/main_storage/production_models/stock_prediction.h5'
    model = keras.models.load_model(model_location)
    return model
