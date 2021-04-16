import os
import sys

whereIam = os.uname()[1]
assert whereIam in [
    "super",
    "wdtim719z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
]

if whereIam in ["super", "wdtim719z"]:
    root = "/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratch_ai4geo/"

if not os.path.exists(root + "miniworld"):
    print("run merge before")
    quit()

if not os.path.exists("build"):
    os.makedirs("build")

if whereIam == "super":
    os.system("/data/anaconda3/bin/python debug.py")
if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python debug.py")
