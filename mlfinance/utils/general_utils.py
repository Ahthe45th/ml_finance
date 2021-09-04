import os
import platform
import datetime
import random
import string
import json
import sys
import traceback

from threading import Thread

threads = {}


def run_thread(targetfunction, argsforfunction=[]):
    """creates a thread for a function using the function name and arguments it may require"""
    threads[targetfunction.__name__] = Thread(
        target=targetfunction, args=argsforfunction
    )
    threads[targetfunction.__name__].start()


def random_filename():
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
    suffix2 = "tempppy"
    patatas = "__".join([suffix, suffix2])
    return patatas


def trace_print_statements():
    class TracePrints(object):
        def __init__(self):
            self.stdout = sys.stdout

        def write(self, s):
            self.stdout.write("Writing %r\n" % s)
            traceback.print_stack(file=self.stdout)

    sys.stdout = TracePrints()


def force_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(name, path, jsonn):
    filename = path + "/" + name + ".json"
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            jsonn = json.dumps(jsonn)
            file.write(jsonn)
        print(f"Saved {name} to {filename}")


def enum_extension_files(dir, extension):
    extension = "." + extension
    filess = []
    for root, dirs, files in os.walk(dir):
        for x in files:
            if x.endswith(extension):
                filess.append(os.path.join(root, x))
    return filess
