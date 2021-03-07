import os
import sys

whereIam = os.uname()[1]
assert whereIam in [
    "super",
    "wdtim719z",
    "ldtis706z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
]

if whereIam in ["super", "wdtim719z"]:
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"

if not os.path.exists(root + "miniworld"):
    print("run merge before")
    quit()

if os.path.exists("build"):
    os.system("rm -rf build")
os.makedirs("build")

if whereIam == "super":
    os.system("/data/anaconda3/bin/python train.py ")
    os.system("/data/anaconda3/bin/python test.py ")
if whereIam in ["ldtis706z","wdtim719z"]:
    os.system("/data/anaconda3/envs/pytorch/bin/python train.py ")
    os.system("/data/anaconda3/envs/pytorch/bin/python test.py ")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    print("TODO")
