import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from mlfinance.utils.enumeration import search_dir

import mlfinance.utils.locales as Location


def dataset_files_enum() -> List:
    """
    This function returns a list of all csv files in the directory
    passed as the first parameter, that have a certain extension
    (the second parameter).
    
    Parameters:
        None
    
    Returns:
        A list of .csv files in 
    """
    domain_to_search = Location.Base()
    files = search_dir(domain_to_search, ".csv")
    return files


def create_dataset(df) -> Tuple:
    """
    Creates the dataset for the model.

    Parameters:
        df (DataFrame): The dataframe of the stock prices.

    Returns:
        x (numpy array): The X set of the data.
        y (numpy array): The Y set of the data.
    """
    # so over here one must realize that the 
    # guy is using the value for the stock 
    # 50 days later as the y
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i - 50 : i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


def dataset_preprocessing():
    """
    Parameters:
        None
    
    Returns:
        - a dictionary containing the preprocessed training and testing sets for each stock.
        - the keys of the dictionary are the stock names, and the values are a list containing:
            - x_train (training set)
            - y_train (training labels)
            - x_test (testing set)
            - y_test (testing labels)
    """
    dataset_files = dataset_files_enum()
    dateset_dict = {}
    for dataset_file in dataset_files:
        stock_name = dataset_file.replace(".csv", "")
        df = pd.read_csv(dataset_file)
        df = df["Open"].values
        df = df.reshape(-1, 1)

        debug_print(df.shape)

        dataset_train = np.array(df[: int(df.shape[0] * 0.8)])
        dataset_test = np.array(df[int(df.shape[0] * 0.8) :])

        debug_print(dataset_train.shape)
        debug_print(dataset_test.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_train = scaler.fit_transform(dataset_train)
        debug_print(dataset_train[:5])

        dataset_test = scaler.transform(dataset_test)
        debug_print(dataset_test[:5])

        x_train, y_train = create_dataset(dataset_train)
        x_test, y_test = create_dataset(dataset_test)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        dataset_dict[stock_name] = [x_train, y_train, x_test, y_test]
    return dataset_dict


def nlp_basic_preprocessing():
    """No need to worry about this rn"""
    pass
