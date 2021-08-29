import pandas as pd
import os

from mlfinance.utils.data.tweets import read_tweets

def tweet_dataset(keyword, stock_hashtag='#stocks', Limit=3500):
    keyword_tweets = read_tweets(keyword, limit=Limit)
    hashtag_tweets = read_tweets(stock_hashtag, limit=Limit)
    all_columns = list(hashtag_tweets.columns.values)
    hashtag_tweets = hashtag_tweets.astype(keyword_tweets.dtypes.to_dict())
    dataset = pd.merge(hashtag_tweets, keyword_tweets, on=all_columns)
    return dataset
