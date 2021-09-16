import os
import sys
import datetime
import random

whereIam = os.uname()[1]
assert whereIam in [
    "wdtim719z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
]

if whereIam == "wdtim719z":
    root = "/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratch_ai4geo/"

if not os.path.exists(root + "miniworld"):
    print("run merge before")
    quit()

if not os.path.exists("build"):
    os.makedirs("build")

today = datetime.date.today()
tmp = random.randint(0, 1000)
myhash = str(today) + "_" + str(tmp)
print(myhash)

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python train_border.py build/" + myhash)
    os.system("/data/anaconda3/envs/pytorch/bin/python test_border.py build/" + myhash)
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python train_border.py build/" + myhash)
    os.system("/d/jcastillo/anaconda3/bin/python test_border.py build/" + myhash)
