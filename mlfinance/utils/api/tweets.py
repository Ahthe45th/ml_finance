'''
Generates tweets about a keyword.
'''
import pandas as pd
import twint
import nest_asyncio
import re

nest_asyncio.apply()

import mlfinance.utils.general.locales as Location
import os

from mlfinance.utils.preprocessing.tweets import read_tweets

def read_tweets(about, limit=1000, time_period="unspecified"):
    """
    Searches for tweests based on either a hashtag or keyword depending on
    whether or not one prefixes keyword with a #
    """
    timestr = time.strftime("%d.%m.%Y")
    c = twint.Config()
    c.Limit = limit
    c.Lang = "en"
    c.Store_csv = True
    c.Search = about
    dir = Location.Home(__file__) + "/" + timestr
    os.makedirs(dir)
    file_locale = dir + "/_" + c.Lang + "_" + about + ".csv"
    c.Output = dir + "/_" + c.Lang + "_" + about + ".csv"
    twint.run.Search(c)
    df = pd.read_csv(file_locale)
    return df
    
def tweet_dataset(keyword, stock_hashtag="#stocks", Limit=3500):
    keyword_tweets = read_tweets(keyword, limit=Limit)
    hashtag_tweets = read_tweets(stock_hashtag, limit=Limit)
    all_columns = list(hashtag_tweets.columns.values)
    hashtag_tweets = hashtag_tweets.astype(keyword_tweets.dtypes.to_dict())
    dataset = pd.merge(hashtag_tweets, keyword_tweets, on=all_columns)
    return dataset
