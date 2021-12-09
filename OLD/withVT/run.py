import os
import sys

whereIam = os.uname()[1]
assert whereIam in [
    "wdtim719z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
    "ldtis706z",
]

if whereIam == "wdtim719z":
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratch_ai4geo/"

if not os.path.exists(root + "miniworld"):
    print("miniworld not found")
    quit()

os.system("rm -rf build")
os.makedirs("build")

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python train.py")
    os.system("/data/anaconda3/envs/pytorch/bin/python test.py")
if whereIam == "ldtis706z":
    os.system("python3 train.py")
    os.system("python3 test.py")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python train.py")
    os.system("/d/jcastillo/anaconda3/bin/python test.py")
