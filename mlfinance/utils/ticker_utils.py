import io
import requests
import pandas as pd
from typing import List, Union

from enum import Enum

_EXCHANGE_LIST = ["nyse", "nasdaq", "amex"]

_SECTORS_LIST = set(
    [
        "Consumer Non-Durables",
        "Capital Goods",
        "Health Care",
        "Energy",
        "Technology",
        "Basic Industries",
        "Finance",
        "Consumer Services",
        "Public Utilities",
        "Miscellaneous",
        "Consumer Durables",
        "Transportation",
    ]
)

headers = {
    "authority": "api.nasdaq.com",
    "accept": "application/json, text/plain, */*",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
    "origin": "https://www.nasdaq.com",
    "sec-fetch-site": "same-site",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
    "referer": "https://www.nasdaq.com/",
    "accept-language": "en-US,en;q=0.9",
}


def params(exchange):
    return (
        ("letter", "0"),
        ("exchange", exchange),
        ("render", "download"),
    )


params = (
    ("tableonly", "true"),
    ("limit", "25"),
    ("offset", "0"),
    ("download", "true"),
)


def params_region(region):
    return (
        ("letter", "0"),
        ("region", region),
        ("render", "download"),
    )


class Region(Enum):
    AFRICA = "AFRICA"
    EUROPE = "EUROPE"
    ASIA = "ASIA"
    AUSTRALIA_SOUTH_PACIFIC = "AUSTRALIA+AND+SOUTH+PACIFIC"
    CARIBBEAN = "CARIBBEAN"
    SOUTH_AMERICA = "SOUTH+AMERICA"
    MIDDLE_EAST = "MIDDLE+EAST"
    NORTH_AMERICA = "NORTH+AMERICA"


class SectorConstants:
    NON_DURABLE_GOODS = "Consumer Non-Durables"
    CAPITAL_GOODS = "Capital Goods"
    HEALTH_CARE = "Health Care"
    ENERGY = "Energy"
    TECH = "Technology"
    BASICS = "Basic Industries"
    FINANCE = "Finance"
    SERVICES = "Consumer Services"
    UTILITIES = "Public Utilities"
    DURABLE_GOODS = "Consumer Durables"
    TRANSPORT = "Transportation"


def get_tickers(
    NYSE: bool = True, NASDAQ: bool = True, AMEX: bool = True, S_AND_P: bool = True
) -> List[str]:
    """
    This function returns a list of tickers in the specified exchanges.

    Parameters:
        NYSE (bool): Whether to include NYSE tickers. Defaults to True.
        NASDAQ (bool): Whether to include NASDAQ tickers. Defaults to True.
        AMEX (bool): Whether to include AMEX tickers. Defaults to True.
        S_AND_P (bool or str): Whether to include S&P 500 tickers. Defaults to True.
            If string, the function will return the difference between the list of
            tickers and the S&P 500 tickers. This is useful when removing tickers
            that are listed in the S&P 500.

    Return:
        list: A list of tickers in the specified exchanges.
    """
    tickers_list = []
    if NYSE:
        tickers_list.extend(__exchange2list("nyse"))
    if NASDAQ:
        tickers_list.extend(__exchange2list("nasdaq"))
    if AMEX:
        tickers_list.extend(__exchange2list("amex"))
    if S_AND_P:
        tickers_list[0:0] = s_and_p_tickers(get_back="Symbol")
    if S_AND_P == "Done":
        tickers_list = [
            x for x in tickers_list if x not in s_and_p_tickers(get_back="Symbol")
        ]
    tickers_list = list(dict.fromkeys(tickers_list))
    return tickers_list


def s_and_p_tickers(get_back: str = "Symbol") -> List[str]:
    """
    This function takes no arguments and returns a list of ticker symbols of companies
    included in S&P 500 index. The data is taken from Wikipedia.

    Arguments:
        get_back (str): This argument specifies what information about companies is returned.
        The default value is "Symbol", which means that just ticker symbols will be returned.
    Returns:
        companies_info (list): A list of ticker symbols or other info about companies.
    """
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    df.to_csv("S&P500-Info.csv")
    companies_info = df[get_back].tolist()
    return companies_info


def get_tickers_filtered(
    mktcap_min: int = None, mktcap_max: int = None, sectors: List[str] = None
) -> List[str]:
    """
    Get tickers from the specified exchanges, filtered by market cap and sectors

    Parameters
    ----------
    mktcap_min : int, optional
        If specified, filter by market cap.
    mktcap_max : int, optional
        If specified, filter by market cap.
    sectors : list of str, optional
        If specified, filter by sectors.

    Returns
    -------
    tickers_list : list of str
        A list of tickers.
    """
    tickers_list = []
    for exchange in _EXCHANGE_LIST:
        tickers_list.extend(
            __exchange2list_filtered(
                exchange, mktcap_min=mktcap_min, mktcap_max=mktcap_max, sectors=sectors
            )
        )
    return tickers_list


def get_biggest_n_tickers(top_n, sectors=None):
    """
    Top N tickers according to market cap.

    Parameters
    ----------
    top_n : int
        The number of tickers to be returned
    sectors : str or list of str, optional
        A string or list of strings of sectors to filter by. Default is None.
        If not None, must be one of the following:
            ['Consumer Discretionary',
            'Consumer Staples',
            'Energy',
            'Financials',
            'Health Care',
            'Industrials',
            'Information Technology',
            'Materials',
            'Real Estate',
            'Telecommunication Services',
            'Utilities']
        If a list is passed, companies will be filtered by all of the passed sectors.
        E.g. passing ['Energy', 'Financials'] will return companies that are in either
        the Energy or Financials sectors.

    Returns
    -------
    tickers : list of str
        A list of tickers from the passed criteria. These are sorted in descending order by market cap.

    Raises
    ------
    ValueError if some invalid sectors are passed.
    ValueError if top_n is larger than the number of companies that meet the other criteria.


    Example usage:  get_biggest_n_tickers(10) returns the 10 biggest companies on the NYSE,
    according to their most recent market cap.
    These are sorted in descending order of market cap.
    The sector of each company is also returned.

        get_biggest_n_tickers(10, sectors=["Energy", "Financials"]) returns the 10 biggest
            companies on the NYSE in the Energy or Financials sectors, according to their
            most recent market cap.

        get_biggest_n_tickers(10, sectors="Utilities") returns the 10 biggest companies on
            the NYSE in the Utilities sector, according to their most recent market cap.

        get_biggest_n_tickers(10, sectors=["Utilities", "Financials"]) returns the 10 biggest
            companies on the NYSE in the Utilities or Financials sectors, according to their
            most recent market cap.

        get_biggest_n_tickers(10, sectors=["Industrials", "Financials", "Utilities"]) returns
            the 10 biggest companies on the NYSE in the Industrials, Financials or Utilities
            sectors, according to their most recent market cap.

        Note that even though Utilities is listed twice above, it will appear only once in this
            list because it is a sector that only has companies with a market cap above
            $100M (as per IEX).

        get_biggest_n_tickers(10, sectors=["Consumer Discretionary", "Financials"]) returns an
            error because Consumer Discretionary is not a valid sector according to IEX.

        get_biggest_n_tickers(10, sectors=["Foo", "Bar"]) returns an error because Foo and Bar
            are not valid sectors according to IEX.
    """
    df = pd.DataFrame()
    for exchange in _EXCHANGE_LIST:
        temp = __exchange2df(exchange)
        df = pd.concat([df, temp])

    df = df.dropna(subset={"marketCap"})
    df = df[~df["Symbol"].str.contains("\.|\^")]

    if sectors is not None:
        if isinstance(sectors, str):
            sectors = [sectors]
        if not _SECTORS_LIST.issuperset(set(sectors)):
            raise ValueError("Some sectors included are invalid")
        sector_filter = df["Sector"].apply(lambda x: x in sectors)
        df = df[sector_filter]

    def cust_filter(mkt_cap):
        if "M" in mkt_cap:
            return float(mkt_cap[1:-1])
        elif "B" in mkt_cap:
            return float(mkt_cap[1:-1]) * 1000
        else:
            return float(mkt_cap[1:]) / 1e6

    df["marketCap"] = df["marketCap"].apply(cust_filter)

    df = df.sort_values("marketCap", ascending=False)
    if top_n > len(df):
        raise ValueError("Not enough companies, please specify a smaller top_n")

    return df.iloc[:top_n]["Symbol"].tolist()


def get_tickers_by_region(region):
    """
    Get a list of tickers from a specific region.

    Parameters
    ----------
    region : Region
        A Region enum value, e.g. Region.ASIA or Region.EUROPE.

    Returns
    -------
    list of str
        A list of tickers from the given region.

    Raises
    ------
    ValueError
        If the region argument is not a valid Region enum value, e.g. Region.AFRICA or Region.ASIA.
    """
    if region in Region:
        response = requests.get(
            "https://old.nasdaq.com/screening/companies-by-name.aspx",
            headers=headers,
            params=params_region(region),
        )
        data = io.StringIO(response.text)
        df = pd.read_csv(data, sep=",")
        return __exchange2list(df)
    else:
        raise ValueError(
            "Please enter a valid region (use a Region.REGION as the argument, e.g. Region.AFRICA)"
        )


def __exchange2df(exchange):
    """
    Parameters
    ----------
    exchange : str
        The name of the stock exchange

    Returns
    -------
    df : DataFrame
        A pandas DataFrame with the stocks listed on the exchange
    """
    r = requests.get(
        "https://api.nasdaq.com/api/screener/stocks", headers=headers, params=params
    )
    data = r.json()["data"]
    df = pd.DataFrame(data["rows"], columns=data["headers"])
    return df


def __exchange2list(exchange):
    df = __exchange2df(exchange)
    df_filtered = df[~df["symbol"].str.contains("\.|\^")]
    return df_filtered["symbol"].tolist()


def __exchange2list_filtered(
    exchange: str,
    mktcap_min: int = None,
    mktcap_max: int = None,
    sectors: Union[str, List[str]] = None,
):
    """
    This function will return a list of tickers from the specified exchange.

    The function accepts the following arguments:

        exchange: string
            The exchange to get the tickers from. Valid values are:
                'amex'
                'nyse'
                'nasdaq'
                'otc'

        mktcap_min: int or float, optional
            Minimum market cap value to filter by. If not specified, no filtering
            will be done.

        mktcap_max: int or float, optional
            Maximum market cap value to filter by. If not specified, no filtering
            will be done.

        sectors: str or list of str, optional
            Sector(s) to filter by. If not specified, no filtering will be done.
            Valid values are:
                'Basic Materials'
                'Conglomerates'
                'Consumer Goods'
                'Financial'
                'Healthcare'
                'Industrial Goods'
                'Services'
                'Technology'
                'Utilities'

        Returns: list of str
            A list of tickers from the specified exchange.

        Example Usage:
        >>> __exchange2list_filtered('nasdaq')
    """
    df = __exchange2df(exchange)
    df = df.dropna(subset={"marketCap"})
    df = df[~df["symbol"].str.contains("\.|\^")]

    if sectors is not None:
        if isinstance(sectors, str):
            sectors = [sectors]
        if not _SECTORS_LIST.issuperset(set(sectors)):
            raise ValueError("Some sectors included are invalid")
        sector_filter = df["sector"].apply(lambda x: x in sectors)
        df = df[sector_filter]

    def cust_filter(mkt_cap):
        if "M" in mkt_cap:
            return float(mkt_cap[1:-1])
        elif "B" in mkt_cap:
            return float(mkt_cap[1:-1]) * 1000
        elif mkt_cap == "":
            return 0.0
        else:
            return float(mkt_cap[1:]) / 1e6

    df["marketCap"] = df["marketCap"].apply(cust_filter)
    if mktcap_min is not None:
        df = df[df["marketCap"] > mktcap_min]
    if mktcap_max is not None:
        df = df[df["marketCap"] < mktcap_max]
    return df["symbol"].tolist()


def save_tickers(
    NYSE: bool = True,
    NASDAQ: bool = True,
    AMEX: bool = True,
    filename: str = "tickers.csv",
) -> None:
    """
    Makes a csv file of the list of tickers from the NYSE, NASDAQ and AMEX.

    Parameters
    ----------
    NYSE : bool, default True
        If True, tickers from NYSE are included.
    NASDAQ : bool, default True
        If True, tickers from NASDAQ are included.
    AMEX : bool, default True
        If True, tickers from AMEX are included.
    filename : str, default "tickers.csv"
        Name of the file to save the tickers.

    Returns
    -------
    None
    """
    tickers2save = get_tickers(NYSE, NASDAQ, AMEX)
    df = pd.DataFrame(tickers2save)
    df.to_csv(filename, header=False, index=False)


def save_tickers_by_region(
    region: str, filename: str = "tickers_by_region.csv"
) -> None:
    """
    Saves a list of tickers that belong to a specified region to a CSV file.

    Parameters:
        region (str): The region to save tickers for.
            Valid regions are:
                AFRICA
                EUROPE
                ASIA
                AUSTRALIA+AND+SOUTH+PACIFIC
                CARIBBEAN
                SOUTH+AMERICA
                MIDDLE+EAST
                NORTH+AMERICA
        filename (str): The name of the file to save the tickers to.

    Returns:
        None
    """
    tickers2save = get_tickers_by_region(region)
    df = pd.DataFrame(tickers2save)
    df.to_csv(filename, header=False, index=False)


if __name__ == "__main__":

    tickers = get_tickers()
    print(tickers[:5])

    tickers = get_tickers(AMEX=False)

    # default filename is tickers.csv, to specify, add argument filename='yourfilename.csv'
    save_tickers()

    # save tickers from NYSE and AMEX only
    save_tickers(NASDAQ=False)

    # get tickers from Asia
    tickers_asia = get_tickers_by_region(Region.ASIA)
    print(tickers_asia[:5])

    # save tickers from Europe
    save_tickers_by_region(Region.EUROPE, filename="EU_tickers.csv")

    # get tickers filtered by market cap (in millions)
    filtered_tickers = get_tickers_filtered(mktcap_min=500, mktcap_max=2000)
    print(filtered_tickers[:5])

    # not setting max will get stocks with $2000 million market cap and up.
    filtered_tickers = get_tickers_filtered(mktcap_min=2000)
    print(filtered_tickers[:5])

    # get tickers filtered by sector
    filtered_by_sector = get_tickers_filtered(
        mktcap_min=200e3, sectors=SectorConstants.FINANCE
    )
    print(filtered_by_sector[:5])

    # get tickers of 5 largest companies by market cap (specify sectors=SECTOR)
    top_5 = get_biggest_n_tickers(5)
    print(top_5)
