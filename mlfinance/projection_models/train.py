import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from mlfinance.projection_models.model_utils import load_torch_model, load_tensorflow_model, save_torch_model, save_tensorflow_model, get_torch_model, get_tensorflow_model
from mlfinance.projection_models.dataset_utils import dataset_preprocessing

def train_tensorflow(x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=50, batch_size=32)

def train_torch(x_train, y_train, epochs, batch_size):
    pass

def prediction_test():
    model = load_model('stock_prediction.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()
