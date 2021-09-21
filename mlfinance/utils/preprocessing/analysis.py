"""

Functions that analyze csv stock data

"""

import yfinance as yf
import pandas as pd
import numpy as np
import finnhub
import shutil
import os
import json
import time
import requests
import datetime

import mlfinance.utils.locales as Location
from mlfinance.utils.general_utils import enum_extension_files, force_mkdir, save_json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sentiment_analyzer = SentimentIntensityAnalyzer()


def gainers(number_to_return=10):
    """
    Should give back the stocks that moved the most from the day before

    Parameters:
        number_to_return (int): The number of stocks to return.

    Returns:
        list: A list of dictionaries, each dictionary containing the name of the stock and the price difference.
    """
    directory = Location.Home(__file__) + "/stocks/short_term"
    csv_files = enum_extension_files(directory, "csv")
    data_frames = [pd.read_csv(file) for file in csv_files]
    price_rises = []
    frame = 0
    for data_frame in data_frames:
        print("-------------")
        print(list(data_frame.columns.values))
        num_rows = data_frame.shape[0]
        for row in range(num_rows):
            row = data_frame.loc[[row]]
            dates = data_frame["Date"]
            open_prices = data_frame["Open"]
            print("Dates:")
            print(dates)
            print("Opening prices:")
            print(open_prices)
            print(row)
            price_diff = open_prices.iloc[-1] - open_prices.iloc[-2]
            print(f"The price diff: {price_diff}")
            name = csv_files[frame]
            stock_name = name.split(".csv")[0].split("/")[-1].replace("_", " ")
            price_rises.append({"stock": stock_name, "price": price_diff})
        frame += 1
    sorted_prices = sorted(price_rises, key=lambda x: x["price"], reverse=True)[
        :number_to_return
    ]
    print(sorted_prices)


def analyze_tweets(tweet_file):
    """
    This function takes a csv file of tweets and returns a list of dictionaries of the form:
        [{'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215},
        {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
        {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
        ...
        ]

    The input file should be a csv file with a column called 'tweet'.
    """
    df = pd.read_csv(tweet_file)
    tweet_list = df.tweet.tolist()
    sentiments = [sentiment_analyzer.polarity_scores(tweet) for tweet in tweet_list]
    return sentiments
