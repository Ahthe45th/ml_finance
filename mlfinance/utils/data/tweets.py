import os
import pandas as pd
import twint
import nest_asyncio
import re

import mlfinance.utils.locales as Location
nest_asyncio.apply()

def read_tweets(about, limit=1000, time_period='unspecified'):
    '''
    Searches for tweests based on either a hashtag or keyword depending on
    whether or not one prefixes keyword with a #
    '''
    timestr = time.strftime("%d.%m.%Y")
    c = twint.Config()
    c.Limit = limit
    c.Lang = "en"
    c.Store_csv = True
    c.Search = about
    dir = Location.Home(__file__) + '/' + timestr
    os.makedirs(dir)
    file_locale = dir + "/_" + c.Lang + "_" + about + ".csv"
    c.Output = dir + "/_" + c.Lang + "_" + about + ".csv"
    twint.run.Search(c)
    df = pd.read_csv(file_locale)
    return df

if __name__ == '__main__':
    keyword = input('Put in keyword: ')
    read_tweets(keyword)
