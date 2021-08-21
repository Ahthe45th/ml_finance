import torch
import os
import pickle as pl
import time



def using_gpu():
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


class Timer(object):
    """
    Useful for timing stuff
    timer = Timer()
    # first process
    timer.start()
    process1()
    timer.stop()
    print(timer.elapsed())
    # second process
    timer.start()
    process2()
    timer.stop()
    print(timer.elapsed())
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