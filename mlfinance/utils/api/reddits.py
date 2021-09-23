# WARNING: We are limited to 60 requests per minute

import praw
import pandas as pd
from praw.models import MoreComments
from bs4 import BeautifulSoup
import requests
from typing import List, Optional, Union
import time
import pickle as pl
import os
from pathed import Path


def pickle(data, path):
    """
    pickle(data, 'path/to/pickled/file')
    """
    with open(path, "wb") as file_handler:
        pl.dump(data, file_handler)


def unpickle(path):
    """
    unpickle('path/to/pickled/file')
    """
    with open(path, "rb") as file_handler:
        return pl.load(file_handler)


def get_page(client_id: str, client_secret: str, url: str) -> "praw.Reddit.Submission":
    """ """
    reddit = praw.Reddit(
        user_agent="Comment Extraction (by /u/Rarest_Cardiologist)",
        client_id=client_id,
        client_secret=client_secret,
    )
    submission = reddit.submission(url=url)
    return submission


def get_posts(client_id: str, client_secret: str, url: str) -> pd.DataFrame:
    """ """
    submission = get_page(client_id=client_id, client_secret=client_secret, url=url)

    posts = []
    for top_level_comment in submission.comments[1:]:
        if isinstance(top_level_comment, MoreComments):
            continue
        posts.append(top_level_comment.body)

    posts = pd.DataFrame(posts, columns=["body"])

    # get rid of removed and deleted comments
    indexNames = posts[(posts.body == "[removed]") | (posts.body == "[deleted]")].index
    posts.drop(indexNames, inplace=True)

    return posts


def get_links_from_search(client_id: str, client_secret: str, url: str) -> List[str]:
    """
    returns all the links that have a comments section
    """
    reddit = praw.Reddit(
        user_agent="Comment Extraction (by /u/Rarest_Cardiologist)",
        client_id=client_id,
        client_secret=client_secret,
    )
    html = reddit.request(method="RAW", path=url).text
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a")
    links = [link.get("href") for link in links if "/comments/" in link.get("href")]
    links = [link for link in links if "https://" in link]
    return links


def make_reddit_api_file(api: dict) -> None:
    """
    stores api in .reddit_api
    """
    file_dir = Path(os.path.dirname(os.path.abspath(__file__)), custom=True)
    pickle(api, file_dir / ".reddit_api")


def read_reddit_api_file() -> dict:
    file_dir = Path(os.path.dirname(os.path.abspath(__file__)), custom=True)
    if os.path.exists(file_dir / ".reddit_api"):
        data = unpickle(file_dir / ".reddit_api")
        return data
    else:
        raise IOError(".reddit_api not found")


def get_reddit_api():
    """
    if .reddit_api file is not found, will ask for client_id and client_secret
    """

    def ask_for_api() -> dict:
        client_id = input("client ID: ")
        client_secret = input("client secret: ")
        api = {"client_id": client_id, "client_secret": client_secret}
        make_reddit_api_file(api)
        return api

    try:
        api = read_reddit_api_file()
        keys = api.keys()
        if ("client_id" not in keys) or ("client_secret") not in keys:
            api = ask_for_api()
        return api
    except IOError:
        api = ask_for_api()
        return api


def get_comments_from_search(
    client_id: str,
    client_secret: str,
    url: str,
    limit: Optional[int] = 3,
    wait_time: str = 2,
) -> Union[List[str], List[pd.DataFrame]]:
    """
    if limit is None, then this will go over all the links
    wait time is the number of seconds that it waits before grabbing the comments on the next link
    """
    api = get_reddit_api()

    links = get_links_from_search(url=url, **api)

    comments = []

    if limit != None:
        iters = min([len(links), limit])
        for i in range(iters):
            link = links[i]
            comments.append(
                get_posts(url=link, **api)
            )
            time.sleep(wait_time)
    else:
        for link in links:
            comments.append(
                get_posts(url=link, **api)
            )
            time.sleep(wait_time)

    return links, comments


def test_get_post(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """ """
    api = get_reddit_api()
    if url is None:
        url = input("url: ")
    return get_posts(url=url, **api)


def test_get_posts_from_search(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """ """
    api=get_reddit_api()
    if url is None:
        url=input("url: ")
    return get_comments_from_search(
        client_id=client_id, client_secret=client_secret, url=url
    )


if __name__ == "__main__":
    print(
        test_get_post(
            url="https://www.reddit.com/r/wallstreetbets/comments/prqy5j/daily_discussion_thread_for_september_20_2021/"
        )
    )
    print(
        test_get_posts_from_search(
            url="https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new%27"
        )
    )
