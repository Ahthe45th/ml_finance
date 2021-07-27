import os
import shutil
import mlfinance.utils.locales as Location

from mlfinance.utils.api_utils import enumerate_stocks_finnhub, enumerate_stocks_yahoo
from mlfinance.utils.ticker_utils import get_tickers
from mlfinance.utils.general_utils import force_mkdir, run_thread

def main():
    if os.path.exists(Location.Home(__file__) + "/stocks/short_term"):
        shutil.rmtree(Location.Home(__file__) + "/stocks/short_term")
    force_mkdir(Location.Home(__file__) + "/stocks/short_term")
    force_mkdir(Location.Home(__file__) + "/stocks/long_term")

    tickers = get_tickers(S_AND_P=True)
    run_thread(enumerate_stocks_finnhub, [tickers])
    Amount_of_API_Calls, Stock_Failure, Stocks_Not_Imported = enumerate_stocks_yahoo(tickers)
    tickers = get_tickers(NYSE=True, NASDAQ=True, AMEX=True, S_AND_P='Done')
    Amount_of_API_Calls, Stock_Failure, Stocks_Not_Imported = enumerate_stocks_yahoo(tickers, Amount_of_API_Calls=Amount_of_API_Calls, Stock_Failure=Stock_Failure, Stocks_Not_Imported=Stocks_Not_Imported)

if __name__ == "__main__":
    main()
