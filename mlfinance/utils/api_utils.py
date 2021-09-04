import os
import re
import json
import time
import shutil
import finnhub
import requests
import datetime
import traceback
import pandas as pd
import yfinance as yf
from googlefinance import getQuotes
import mlfinance.utils.locales as Location
from typing import Dict

from mlfinance.utils.ticker_utils import get_tickers, s_and_p_tickers
from mlfinance.utils.general_utils import enum_extension_files, force_mkdir, save_json

s_and_p_names = s_and_p_tickers(get_back="Security")
s_and_p_symbols = s_and_p_tickers()
s_and_p_dict = {
    ticker: s_and_p_names[s_and_p_symbols.index(ticker)] for ticker in s_and_p_symbols
}

finnhub_api_key = "c3ctr3qad3i868dolgpg"
finnhub_client = finnhub.Client(api_key=finnhub_api_key)


def finnhub_metrics(ticker: str) -> None:
    """
    This function is a wrapper for Finnhub's API.
    It takes in a ticker, and saves the following data to a folder:

    stock_candles:
        - date
        - open
        - high
        - low
        - close
        - volume
    aggregate_indicator:
        - date
        - macd
        - macds
        - macdh
        - slowk
        - slowd
    basic_financials:
        - date
        - grossmargin
        - profitmargin
        - operatingmargin
        - currentratio
        - debt2asset
        - quickratio
        - epstttm
        - epslatest
    earnings:
        - date
        - actual
        - estimate
    news:
        - datetime
        - headline
        - source
        - url

    The data is saved to a folder with the following structure:
    stocks/finnhub_metrics/{ticker}/{last_year_str}_{today_str}/stock_candles.json
    stocks/finnhub_metrics/{ticker}/{last_year_str}_{today_str}/aggregate_indicator.json
    stocks/finnhub_metrics/{ticker}/{last_year_str}_{today_str}/basic_financials.json
    stocks/finnhub_metrics/{ticker}/{last_year_str}_{today_str}/earnings.json
    stocks/finnhub_metrics/{ticker}/{last_year_str}_{today_str}/news.json

    Parameters:
        ticker: A string representing the ticker of the stock.

    Returns:
        None. Saves data to a file.
    """
    metrics_pth = Location.Home(__file__) + "/stocks/finnhub_metrics/"
    ticker_pth = metrics_pth + ticker
    today_date = datetime.datetime.today().replace(day=1)
    last_year_today_date = today_date - datetime.timedelta(days=365)
    today_timestamp = int(today_date.timestamp())
    last_year_timestamp = int(last_year_today_date.timestamp())
    today_str = today_date.strftime("%Y-%m-%d")
    last_year_str = last_year_today_date.strftime("%Y-%m-%d")
    year_pth = ticker_pth + "/" + last_year_str + "_" + today_str
    force_mkdir(year_pth)
    stock_candles = finnhub_client.stock_candles(
        ticker, "D", last_year_timestamp, today_timestamp
    )
    save_json("stock_candles", year_pth, stock_candles)
    aggregate_indicator = finnhub_client.aggregate_indicator(ticker, "D")
    save_json("aggregate_indicator", year_pth, aggregate_indicator)
    basic_financials = finnhub_client.company_basic_financials(ticker, "margin")
    save_json("basic_financials", year_pth, basic_financials)
    earnings = finnhub_client.company_earnings(ticker, limit=5)
    save_json("earnings", year_pth, earnings)
    news = finnhub_client.company_news(ticker, _from=last_year_str, to=today_str)
    save_json("news", year_pth, news)


def fetchstockquotes_google(symbol: str) -> Dict:
    """
    Fetches stock quotes from Google Finance.

    Parameters:
        symbol: A string representing the stock symbol to fetch a quote for.

    Returns:
        A dictionary containing the stock quote.
    """
    while True:
        time.sleep(5)
        os.system("cls" if os.name == "nt" else "clear")
        return getQuotes(symbol)


def get_price(tickers: str) -> Dict:
    """
    WARNING: NOT YET IMPLEMENTED

    Returns the historical stock prices for the given tickers.

    Args:
        tickers (list): A list of stock tickers.

    Returns:
        dict: A dictionary of stock tickers and their historical prices.
    """
    url = (
        "https://financialmodelingprep.com/api/v3/historical-price-full/MSFT,AAPL,GOOG"
    )
    session = requests.session()
    request = session.get(url, timeout=15)
    stock_data = request.json()
    return stock_data


def get_company_name(symbol):
    """
    This function takes a stock symbol as input and returns the name of the company that
    owns the stock.

    Parameters:
        symbol (str): The stock symbol for a publicly traded company.

    Returns:
        str: The name of the company that owns the stock.

    Examples:
        >>> get_company_name("MSFT")
        'Microsoft Corporation'

        >>> get_company_name("AAPL")
        'Apple Inc.'
    """
    if symbol in s_and_p_dict:
        return s_and_p_dict[symbol]
    else:
        url = f"http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={symbol}&region=1&lang=en"
        result = requests.get(url).json()
        for item in result["ResultSet"]["Result"]:
            if item["symbol"] == symbol:
                return item["name"]


def get_historical_data(ticker, period="max"):
    """
    Get historical data for a stock from yahoo finance.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    period : str, optional
        The period of time to get data for. Default is 'max'.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the historical data for the stock.
    """
    today_date = datetime.date.today().replace(day=1)
    last_year_today_date = today_date - datetime.timedelta(days=365)
    today_str = today_date.strftime("%Y-%m-%d")
    last_year_str = last_year_today_date.strftime("%Y-%m-%d")
    year_pth = (
        Location.Home(__file__) + "/stocks/long_term/" + today_str + "_" + last_year_str
    )
    if not os.path.exists(year_pth):
        force_mkdir(year_pth)
        ticker_pth = year_pth + "/" + ticker + ".csv"
        if not os.path.exists(ticker_pth):
            if period == "max":
                temp = yf.Ticker(ticker)
                data = temp.history(period="max")
                data.to_csv(ticker_pth)
            else:
                data = yf.download(ticker, start=last_year_str, end=today_str)
                data.to_csv(ticker_pth)


def enumerate_stocks_yahoo(
    tickers, Amount_of_API_Calls=0, Stock_Failure=0, Stocks_Not_Imported=0
):
    """
    This function will import all of the stocks from a list of tickers.
    It will use the yfinance API to import the data, and it will use
    the time module to wait 5 seconds between each call to yfinance.
    It will also wait for 1 hour if the amount of API calls exceeds 800.
    The function will return the number of API calls made, the number of
    stocks that couldn't be imported, and the total number of stocks
    that were attempted to be imported.

    Parameters:
        tickers (list): A list of tickers that we want to import.

    Returns:
        Amount_of_API_Calls (int): The amount of API calls that were made.
        Stock_Failure (int): The amount of stocks that failed to be imported.
        Stocks_Not_Imported (int): The amount of stocks that weren't imported.

    Example:
        >>> enumerate_stocks_yahoo(['TSLA', 'AAPL'])
        The amount of stocks chosen to observe: 2
        TSLA stock imported successfully for dates:2019-03-28-2019-04-03. Imported 1 stocks and made 1
        AAPL stock imported successfully for dates:2019-03-28-2019-04-03. Imported 1 stocks and made 2
        The amount of stocks we successfully imported:0

        >>> enumerate_stocks_yahoo(['TSLA', 'AAPL', 'GOOG'])
        The amount of stocks chosen to observe: 3
        TSLA stock imported successfully for dates:2019-03-28-2019-04-03. Imported 1 stocks and made 3
        AAPL stock imported successfully for dates:2019-03-28-2019-04-03. Imported 1 stocks and made 4
        GOOG stock imported successfully for dates:2019-03-28-2019-04-03. Imported 1 stocks and made 5
        The amount of stocks we successfully imported:2
    """
    print("The amount of stocks chosen to observe: " + str(len(tickers)))
    for ticker in tickers:
        if Amount_of_API_Calls < 800:
            try:
                name = get_company_name(ticker).replace(" ", "_")
                get_historical_data(ticker)
                # Instantiate the ticker as a stock with Yahoo Finance
                temp = yf.Ticker(ticker)
                # Tells yfinance what kind of data we want about this stock (In this example, all of the historical data)
                Hist_data = temp.history(period="5d")
                print(Hist_data)
                Hist_data.to_csv(
                    Location.Home(__file__)
                    + "/stocks/short_term/"
                    + stock
                    + "_"
                    + name
                    + ".csv"
                )
                # Pauses the loop for 5 seconds so we don't cause issues with Yahoo Finance's backend operations
                time.sleep(5)
                Amount_of_API_Calls += 1
                Stock_Failure = 0
                print(
                    f"{name} stock imported successfully for dates:{first_date}-{second_date}. Imported {i} stocks and made {Amount_of_API_Calls}"
                )
            except ValueError:
                print("Yahoo Finance Back-end Error, Attempting to Fix")
                # Move on to the next ticker if the current ticker fails more than 5 times
                print(traceback.format_exc())
                if Stock_Failure > 5:
                    Stocks_Not_Imported += 1
                if Stocks_Not_Imported >= 15:
                    print(
                        "There seems to be a rather serious problem on the yahoo back end. Look into it. Stopping for now"
                    )
                    break
                Amount_of_API_Calls += 1
                Stock_Failure += 1
        else:
            Amount_of_API_Calls = 0
            for second in range(60 * 60):
                time.sleep(1)
                by_5 = second % 5
                if by_5 == 0:
                    print(f"Waited {time_waited} seconds")
    print(f"The amount of stocks we successfully imported:{i - Stocks_Not_Imported}")
    return Amount_of_API_Calls, Stock_Failure, Stocks_Not_Imported


def enumerate_stocks_finnhub(tickers):
    """
    This function takes a list of ticker strings and returns a dictionary
    with the tickers as keys and a dictionary of their financial metrics as
    values. This function makes use of the Finnhub API to obtain the data.

    Args:
        tickers (list): A list of ticker strings.

    Returns:
        finnhub_dict (dict): A dictionary with the tickers as keys and a
        dictionary of financial metrics as values.

    Example:
        >>> tickers = ['aapl', 'msft', 'amzn']
        >>> finnhub_dict = enumerate_stocks_finnhub(tickers)
        >>> print(finnhub_dict)
        {'aapl': {'name': 'Apple Inc.', 'price': ..., ...},
        'msft': {'name': 'Microsoft Corporation', 'price': ..., ...},
        'amzn': {'name': 'Amazon.com, Inc.', 'price': ..., ...}}
    """
    num_of_calls = 0
    for ticker in tickers:
        num_of_calls += 5
        print(f"Number of calls made to the API: {num_of_calls}")
        if num_of_calls >= 14:
            print("Pausing enumeration")
            time.sleep(60)
            print("Resuming enumeration")
            num_of_calls = 0
        else:
            finnhub_metrics(ticker)
