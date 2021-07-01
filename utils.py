import os
import logging
import datetime
import random
import string
import json
import sys
import traceback

def silence_tensorflow():
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def random_filename():
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
    suffix2 = 'tempppy'
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
    dirstring = '/'
    dirs = path.split('/')
    for dir in dirs:
        dir_rn = dirstring + dir + '/'
        if not os.path.exists(dirstring + dir):
            os.mkdir(dir_rn)

def save_json(name, path, jsonn):
    filename = path + '/' + name + '.json'
    with open(filename, 'w') as file:
        jsonn = json.dumps(jsonn)
        file.write(jsonn)


def enum_extension_files(dir, extension):
    extension = '.' + extension
    filess = []
    for root, dirs, files in os.walk(dir):
        for x in files:
            if x.endswith(extension):
                filess.append(os.path.join(root, x))
    return filess
