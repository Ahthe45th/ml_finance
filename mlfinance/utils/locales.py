import os

def Self(file):
    '''Gets the home path for the file called in'''
    location = os.path.realpath(file)
    return location

def Home(file):
    '''Gets the home path for the file called in'''
    home = os.path.dirname(os.path.realpath(file))
    return home

def Base():
    '''returns projects first dir '''
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
