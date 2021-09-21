import torch
import os
import pickle as pl
import time


def using_gpu():
    """ """
    try:
        torch.cuda.current_device()
        # GPU is being used!
        return True
    except AssertionError:
        # GPU is not being used!
        return False
    except AttributeError:
        # GPU is not being used!
        return False
    except RuntimeError:
        # GPU is not being used!
        return False


def pickle(data, path):
    """
    One-liner for pickling data

    pickle(data, 'path/to/pickled/file')
    """
    with open(path, "wb") as file_handler:
        pl.dump(data, file_handler)


def unpickle(path):
    """
    One-liner for unpickling data

    unpickle('path/to/pickled/file')
    """
    with open(path, "rb") as file_handler:
        return pl.load(file_handler)


class Timer(object):
    """
    Timer for timing stuff

    Example
    -------
    >>> timer = Timer()
    >>> timer.start()
    >>> process1()
    >>> timer.stop()
    8.535007953643799
    >>> print(timer.elapsed())
    8.535007953643799
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        print(self.elapsed())

    def elapsed(self):
        return self.end_time - self.start_time
