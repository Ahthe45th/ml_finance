import requests
import nltk
import warnings
import time

warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup

from urllib.parse import urlencode

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_articles(searchterm):
    '''Gets info on articles related to a search term'''
    query = urlencode({'q': f'Searchterm "{searchterm}"'})
    url = "https://news.google.com/rss/search?" + query
    # make requests
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')
    articles = soup.findAll('item')
    articles_dicts = [{'title':a.find('title').text,'link':a.link.next_sibling.replace('\n','').replace('\t',''),'description':a.find('description').text,'pubdate':a.find('pubdate').text} for a in articles]
    urls = [d['link'] for d in articles_dicts if 'link' in d]
    titles = [d['title'] for d in articles_dicts if 'title' in d]
    descriptions = [d['description'] for d in articles_dicts if 'description' in d]
    pub_dates = [d['pubdate'] for d in articles_dicts if 'pubdate' in d]
    return pub_dates, urls, descriptions, titles

def saferequest(url, delay):
    try:
        if delay:
            time.sleep(delay)
        resp = requests.get(url)
        return resp
    except:
        print(f'Failed at url:{url}')
        return False

def get_articles_sentiment(searchterm, delay=False):
    '''Gets sentiments, urls and dates for searchterm and implements a delay if wanted'''
    dates, urls, description, titles = get_articles(searchterm)
    sentiments = []
    for url in urls:
        resp = saferequest(url, delay)
        if resp:
            soup = BeautifulSoup(resp.text, 'lxml')
            sentences = soup.findAll("p")
            passage = ""
            for sentence in sentences:
                passage += sentence.text
                sentiment = sia.polarity_scores(passage)['compound']
                print(sentiment)
                sentiments.append(sentiment)
    return dates, sentiments, urls
