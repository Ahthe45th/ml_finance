"""

imports util functions for convenience

"""


import sys
from getpaths import getpath

sys.path.append(getpath() / ".." / "utils")

from import_all_utils import *

sys.path.remove(getpath() / ".." / "utils")