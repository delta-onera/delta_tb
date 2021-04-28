import os
import sys
import datetime
import random

whereIam = os.uname()[1]
assert whereIam in [
    "super",
    "wdtim719z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
]

if whereIam == "super":
    os.system("/data/anaconda3/bin/python test.py " + sys.argv[1])
if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python test.py " + sys.argv[1])
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python test.py " + sys.argv[1])
