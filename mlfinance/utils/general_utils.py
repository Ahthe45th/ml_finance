import os
import platform
import datetime
import random
import string
import json
import sys
import traceback
from typing import Dict

from threading import Thread

threads = {}


def run_thread(targetfunction, argsforfunction=[]):
    """
    A function that will run a function in a thread by calling the targetfunction.__name__
        with the targetfunction and any args for the targetfunction as argsforfunction.
    Returns a dictionary of threads with a key based on the function's name.
    Threads will also have a daemon active.

    params:
        targetfunction: The function to be ran in a thread.  Must have a __name__ attribute.
        argsforfunction: A list containing any arguments required by the passed function.
                         Defaults to an empty list.
                         Can contain nested lists of args if they are required for multiple levels
                            of nested functions.

    returns:
        A dictionary containing a key based on the function name and a thread as a value with a
            daemon active.

    example:
        def examplefunc():
            print("example function")

        run_thread(examplefunc)

    result:
        A dictionary containing the following items:
            ('examplefunc'): <Thread(Thread-1, started 140275627439584)>

    notes:
        To pass args as a list to as an arg for as function as argsforfunction modify as follows:

        def examplefunc(*args):
    """
    threads[targetfunction.__name__] = Thread(
        target=targetfunction, args=argsforfunction
    )
    threads[targetfunction.__name__].start()


def random_filename():
    """
    A function that generates random str filenames for testing purposes.

    Args:
        None

    Returns:
        patatas (str): A string containing the concatenation of two strings and a float
    """
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
    suffix2 = "tempppy"
    patatas = "__".join([suffix, suffix2])
    return patatas


def trace_print_statements():
    """
    Decorator for printing out all calls to a function, along with their arguments.
    This is useful for inspecting the exact parameters that are being supplied to a function.

    E.g.:

        >>> @trace_print_statements()
        ... def test(*args, **kwargs): test(*args, **kwargs)
        >>> test(1,2,3)
        Writing 'call test(*args, **kwargs) args=((1, 2, 3),) kwargs={}'
        call test(*args, **kwargs) args=((1, 2, 3),) kwargs={}
        BACK <module> line 28 test(1, 2, 3)
        BACK <module> line 22 test(1, 2, 3)
        call trace_print_statements() args=() kwargs={}
        BACK <module> line 4 TracePrints.__init__ self
        ...
    """

    class TracePrints(object):
        def __init__(self):
            self.stdout = sys.stdout

        def write(self, s):
            self.stdout.write("Writing %r\n" % s)
            traceback.print_stack(file=self.stdout)

    sys.stdout = TracePrints()


def force_mkdir(path):
    """
    makes a directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(name: str, path: str, jsonn: Dict) -> None:
    """
    saves json file to path/name.json

    Parameters:
        name: str
        path: str
        jsonn: dict

    Returns:
        None
    """
    filename = path + "/" + name + ".json"
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            jsonn = json.dumps(jsonn)
            file.write(jsonn)
        print(f"Saved {name} to {filename}")


def enum_extension_files(dir, extension):
    """
    Generate a list of files with a particular extension
    in a directory and all its subdirectories.

    works exactly like glob.glob('*.ext')
    """
    extension = "." + extension
    filess = []
    for root, dirs, files in os.walk(dir):
        for x in files:
            if x.endswith(extension):
                filess.append(os.path.join(root, x))
    return filess
