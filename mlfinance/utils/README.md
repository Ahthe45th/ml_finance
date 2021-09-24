API tools for stock data found here 🛠

# Overview
```yaml
utils:

 - api/reddits.py: download comments on posts from Reddit

 - api/stocks.py: download stock data using finnhub,
                  fetch stock quotes from google finance and yahoo finance

 - api/tickers.py: functions for getting tickers (tickers are the is just another
                   name for change in price) from NYSE, NASDAQ, AMEX, and S&P500

 - api/tickers.py: functions for getting tweets from (you'll never guess) Twitter

 - general/convenience_functions.py: check if gpu is being used, timer for timing objects,
                                     pickling and unpickling objects

 - general/enumeration.py: recursively search through folders to find files with a certain
                           extension, exactly like glob.glob('**.ext')

 - general/general_functions.py: has function to run other functions in its own thread, can be
                                 paired with asyncio for multithreading with semaphores

 - general/locales.py: get absolute path to ml_finance folder and mlfinance.utils folder

 - import_all_utils.py: legacy code for importing all utilities

 - preprocessing/analysis.py: analyze csv stock data
```

# How to Use
- All the functions, classes and objects found in the utils folder are generally to be imported
  into other regions of the project for use there. 
