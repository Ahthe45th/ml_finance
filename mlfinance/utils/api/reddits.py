# WARNING: We are limited to 60 requests per minute per account, until we get the API key

import os
import time
import praw
import requests
import pickle as pl
import pandas as pd
from pathed import Path
from bs4 import BeautifulSoup
from praw.models import MoreComments
from typing import List, Optional, Union, Any, Tuple


def pickle(data: Any, path: str) -> None:
    """
    pickles data

    Parameters
    ----------
    data: Any
        Any python object
    path : str
        File path to save the object in
        
    Returns
    -------
    Nothing
    
    Examples
    --------
    >>> pickle({'real':'dictionary'}, '/path/to/dictionary')
    """
    with open(path, "wb") as file_handler:
        pl.dump(data, file_handler)


def unpickle(path) -> Any:
    """
    returns unpickled objects

    Parameters
    ----------
    path : str
        A path to pickle file
        
    Returns
    -------
    object : Any
        Any python file that was pickled.
    
    Examples
    --------
    >>> Dict = unpickle('/path/to/dictionary')
    {'real':'dictionary'}
    """
    with open(path, "rb") as file_handler:
        return pl.load(file_handler)


def get_page(url: str, reddit: Any = "praw.Reddit") -> "praw.Reddit.Submission":
    """
    gets reddit page
    """
    submission = reddit.submission(url=url)
    return submission


def get_posts(url: str, reddit: Any = "praw.Reddit") -> pd.DataFrame:
    """
    This function takes a url to a reddit post and returns a dataframe of the top level comments.
    
    Parameters
    ----------
    url : str
        A url to a reddit post.
    reddit : Any
        A praw reddit object.
        
    Returns
    -------
    posts : pd.DataFrame
        A dataframe of the top level comments.
    """
    submission = get_page(url=url, reddit=reddit)

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


def get_links_from_search(url: str, reddit: Any = "praw.Reddit") -> List[str]:
    """
    Returns a list of links from a reddit search result page.
    
    Parameters
    ----------
    url : str
        The URL of the search result page.
    reddit : Any
        The Reddit instance to use.
        
    Returns
    -------
    List[str]
        A list of links from the search result page.
    """
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
    """
    retrieves reddit api file
    """
    file_dir = Path(os.path.dirname(os.path.abspath(__file__)), custom=True)
    if os.path.exists(file_dir / ".reddit_api"):
        data = unpickle(file_dir / ".reddit_api")
        return data
    else:
        raise IOError(".reddit_api not found")


def get_reddit_api() -> "praw.Reddit":
    """
    Returns a praw.Reddit object, which is a connection to the Reddit API.
    The user must have a Reddit account and register an app at
    https://www.reddit.com/prefs/apps/.

    The user must provide their client ID and client secret in the following
    format:

        client_id = "XXXXXXXXXXXXXXXXXXXXX"
        client_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    The user can either provide these credentials now or else provide them in
    a file named reddit_api.txt in the same directory as this file. The file
    should contain the two lines above, with each variable on its own line.

    If the user provides the credentials now, then they will be saved in a file
    called reddit_api.txt in the same directory as this file. If the user has
    already provided credentials before, then the credentials will be read from
    the file.

    If the user has not provided credentials and does not have a file called
    reddit_api.txt, then the function will ask the user for their credentials,
    save them to a file called reddit_api.txt, and return a praw.Reddit object.

    If the user has not provided credentials and does have a file called
    reddit_api.txt, then the function will read the credentials from the file
    and return a praw.Reddit object.

    Returns:
        A praw.Reddit object, which is used to access the Reddit API.
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
    except IOError:
        api = ask_for_api()

    reddit = praw.Reddit(
        user_agent="Comment Extraction (by /u/The_Hobbit)",
        **api,
    )

    return reddit


def get_comments_from_search(
    url: str,
    reddit: Any = "praw.Reddit",
    limit: Optional[int] = 3,
    wait_time: str = 2,
) -> Tuple[List[str], List[pd.DataFrame]]:
    """
    This function takes a url and returns a list urls to comments from the latest posts
    from that subreddit. If limit is None, it scrapes all the comments from all posts.

    Parameters
    ----------
    url : str
        The url of the subreddit to scrape.
    reddit : Any, optional
        The reddit api to use. The default is "praw.Reddit".
    limit : Optional[int], optional
        The number of posts to scrape. The default is 3.
    wait_time : str, optional
        The time to wait between requests. The default is 2.

    Returns
    -------
    Tuple[List[str], List[pd.DataFrame]]
        1. A list of urls for each post in the subreddit.
        2. A list of comments from each posts from that subreddit.

    Examples
    --------
    >>> get_comments_from_search(url="https://www.reddit.com/r/learnpython/")
    [
        '/r/learnpython/comments/d7fi0b/python_equivalent_of_phps_var_dump/', 
        '/r/learnpython/comments/d7fi0b/python_equivalent_of_java_var_dump/',
        ...
    ], [
                                                        body
        0    Man, good thing those FED officials sold every...
        1    The Fed presidents disclosed they sold their s...
        ...
        [474 rows x 1 columns],
                                                        body
        0    So, Lucid market cap is $39B without even sell...
        1    "Evergrande, the world's most indebted develop...
        ...
        [474 rows x 1 columns],
        ...
    ]

    """
    reddit = get_reddit_api()

    links = get_links_from_search(url=url, reddit=reddit)

    comments = []

    if limit != None:
        iters = min([len(links), limit])
        for i in range(iters):
            link = links[i]
            comments.append(get_posts(url=link, reddit=reddit))
            time.sleep(wait_time)
    else:
        for link in links:
            comments.append(get_posts(url=link, reddit=reddit))
            time.sleep(wait_time)

    return links, comments


def test_get_post(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get comments from a reddit post.

    Parameters
    ----------
    client_id: str, optional
        The client id for the reddit api.
    client_secret: str, optional
        The client secret for the reddit api.
    url: str, optional
        The url of the reddit post.

    Returns
    -------
    pd.DataFrame
        The comments from the reddit post.

    Examples
    --------
    >>> test_get_posts_from_search()
    url: https://www.reddit.com/r/learnpython/comments/dw53wt/how_to_get_data_from_a_website_that_doesnt_give/
                                                      body
    0    I have been working on a project where I need ...
    1    This is another fake comment. I will now tell ...
    ...
    [474 rows x 1 columns]
    """
    reddit = get_reddit_api()
    if url is None:
        url = input("url: ")
    return get_posts(url=url, reddit=reddit)


def test_get_posts_from_search(
    url: Optional[str] = None,
) -> Tuple[List[str], List[pd.DataFrame]]:
    """
    Get comments from multiple posts.

    Parameters
    ----------
    url: str, optional
        URL to a reddit search

    Returns
    -------
    ([url, url, ...], [pd.DataFrame, pd.DataFrame, ...])
        The url for each post and the comments for each url, respectively.
    """
    reddit = get_reddit_api()
    if url is None:
        url = input("url: ")
    return get_comments_from_search(url=url, reddit=reddit)


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
