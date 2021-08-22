import yfinance as yf
import pandas as pd
import finnhub
import shutil
import traceback
import os
import json
import time
import requests
import datetime
import twint
import nest_asyncio
nest_asyncio.apply()
import time
import pandas as pd
import os
import re


import mlfinance.utils.locales as Location

from mlfinance.utils.ticker_utils import get_tickers, s_and_p_tickers
from mlfinance.utils.general_utils import enum_extension_files, force_mkdir, save_json

from googlefinance import getQuotes



s_and_p_names = s_and_p_tickers(get_back='Security')
s_and_p_symbols = s_and_p_tickers()
s_and_p_dict = {ticker:s_and_p_names[s_and_p_symbols.index(ticker)] for ticker in s_and_p_symbols}

finnhub_api_key = 'c3ctr3qad3i868dolgpg'
finnhub_client = finnhub.Client(api_key=finnhub_api_key)



def finnhub_metrics(ticker):
    metrics_pth = Location.Home(__file__) + '/stocks/finnhub_metrics/'
    ticker_pth = metrics_pth + ticker
    today_date = datetime.datetime.today().replace(day=1)
    last_year_today_date = today_date - datetime.timedelta(days=365)
    today_timestamp = int(today_date.timestamp())
    last_year_timestamp = int(last_year_today_date.timestamp())
    today_str = today_date.strftime("%Y-%m-%d")
    last_year_str = last_year_today_date.strftime("%Y-%m-%d")
    year_pth = ticker_pth + '/' + last_year_str + '_' + today_str
    force_mkdir(year_pth)
    stock_candles = finnhub_client.stock_candles(ticker, 'D', last_year_timestamp, today_timestamp)
    save_json('stock_candles', year_pth, stock_candles)
    aggregate_indicator = finnhub_client.aggregate_indicator(ticker, 'D')
    save_json('aggregate_indicator', year_pth, aggregate_indicator)
    basic_financials = finnhub_client.company_basic_financials(ticker, 'margin')
    save_json('basic_financials', year_pth, basic_financials)
    earnings = finnhub_client.company_earnings(ticker, limit=5)
    save_json('earnings', year_pth, earnings)
    # eps_estimates = finnhub_client.company_eps_estimates(ticker, freq='quarterly')
    # save_json('eps_estimates', year_pth, eps_estimates)
    # execs = finnhub_client.company_executive(ticker)
    # save_json('execs', year_pth, execs)
    news = finnhub_client.company_news(ticker, _from=last_year_str, to=today_str)
    save_json('news', year_pth, news)



def fetchstockquotes_google(symbol):
    while True:
        time.sleep(5)
        os.system('cls' if os.name=='nt' else 'clear')
        return getQuotes(symbol)



def get_price(tickers):
    url = "https://financialmodelingprep.com/api/v3/historical-price-full/MSFT,AAPL,GOOG"
    session = requests.session()
    request = session.get(url, timeout=15)
    stock_data = request.json()
    return stock_data



def get_company_name(symbol):
    if symbol in s_and_p_dict:
        return s_and_p_dict[symbol]
    else:
        url = f"http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={symbol}&region=1&lang=en"
        result = requests.get(url).json()
        for item in result['ResultSet']['Result']:
            if item['symbol'] == symbol:
                return item['name']



def get_historical_data(ticker, period='max'):
    today_date = datetime.date.today().replace(day=1)
    last_year_today_date = today_date - datetime.timedelta(days=365)
    today_str = today_date.strftime("%Y-%m-%d")
    last_year_str = last_year_today_date.strftime("%Y-%m-%d")
    year_pth = Location.Home(__file__) + '/stocks/long_term/' + today_str + '_' + last_year_str
    if not os.path.exists(year_pth):
        force_mkdir(year_pth)
        ticker_pth = year_pth + '/' + ticker + '.csv'
        if not os.path.exists(ticker_pth):
            if period == 'max':
                temp = yf.Ticker(ticker)
                data = temp.history(period='max')
                data.to_csv(ticker_pth)
            else:
                data = yf.download(ticker, start=last_year_str, end=today_str)
                data.to_csv(ticker_pth)



def enumerate_stocks_yahoo(tickers, Amount_of_API_Calls=0, Stock_Failure=0, Stocks_Not_Imported=0):
    print("The amount of stocks chosen to observe: " + str(len(tickers)))
    for ticker in tickers:
        if Amount_of_API_Calls < 800:
            try:
                name = get_company_name(ticker).replace(' ', '_')
                get_historical_data(ticker)
                # Instantiate the ticker as a stock with Yahoo Finance
                temp = yf.Ticker(ticker)
                # Tells yfinance what kind of data we want about this stock (In this example, all of the historical data)
                Hist_data = temp.history(period="5d")
                print(Hist_data)
                Hist_data.to_csv(Location.Home(__file__) + '/stocks/short_term/' + stock + '_' + name + '.csv')
                # Pauses the loop for 5 seconds so we don't cause issues with Yahoo Finance's backend operations
                time.sleep(5)
                Amount_of_API_Calls += 1
                Stock_Failure = 0
                print(f'{name} stock imported successfully for dates:{first_date}-{second_date}. Imported {i} stocks and made {Amount_of_API_Calls}')
            except ValueError:
                print("Yahoo Finance Back-end Error, Attempting to Fix")
                # Move on to the next ticker if the current ticker fails more than 5 times
                print(traceback.format_exc())
                if Stock_Failure > 5:
                    Stocks_Not_Imported += 1
                if Stocks_Not_Imported >= 15:
                    print('There seems to be a rather serious problem on the yahoo back end. Look into it. Stopping for now')
                    break
                Amount_of_API_Calls += 1
                Stock_Failure += 1
        else:
            Amount_of_API_Calls = 0
            for second in range(60*60):
                time.sleep(1)
                by_5 = second % 5
                if by_5 == 0:
                    print(f'Waited {time_waited} seconds')
    print(f"The amount of stocks we successfully imported:{i - Stocks_Not_Imported}")
    return Amount_of_API_Calls, Stock_Failure, Stocks_Not_Imported



def enumerate_stocks_finnhub(tickers):
    num_of_calls = 0
    for ticker in tickers:
        num_of_calls += 5
        print(f'Number of calls made to the API: {num_of_calls}')
        if num_of_calls >= 14:
            print('Pausing enumeration')
            time.sleep(60)
            print('Resuming enumeration')
            num_of_calls = 0
        else:
            finnhub_metrics(ticker)



def read_tweets(about, limit=1000):
    timestr = time.strftime("%d.%m.%Y")
    c = twint.Config()
    c.Limit = limit
    c.Lang = "en"
    c.Store_csv = True
    c.Search = about
<<<<<<< HEAD
    dir = Location.Home(__file__) + '/' + timestr
=======
    dir = Location.Base() + '/' + timestr
>>>>>>> 22998ebf8b4b897ea1cb3b794cb94c88fb007676
    os.makedirs(dir)
    c.Output = dir + "/_" + c.Lang + "_" + about + ".csv"
    twint.run.Search(c)

<<<<<<< HEAD
=======

>>>>>>> 22998ebf8b4b897ea1cb3b794cb94c88fb007676
if __name__ == '__main__':
    keyword = input('Put in keyword: ')
    read_tweets(keyword)
