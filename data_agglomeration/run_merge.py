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

if os.path.exists(root + "miniworld"):
    print("it seems miniworld exists, please remove it by hand")
    quit()

os.makedirs(root + "miniworld")
if whereIam == "super":
    os.makedirs(root + "miniworld/toulouse")
    os.makedirs(root + "miniworld/toulouse/train")
    os.makedirs(root + "miniworld/toulouse/test")
    os.makedirs(root + "miniworld/potsdam")
    os.makedirs(root + "miniworld/potsdam/train")
    os.makedirs(root + "miniworld/potsdam/test")

    os.system("/data/anaconda3/envs/rahh/bin/python merge_mini_world.py ")
if whereIam == "ldtis706z":
    os.makedirs(root + "miniworld/toulouse")
    os.makedirs(root + "miniworld/toulouse/train")
    os.makedirs(root + "miniworld/toulouse/test")
    os.makedirs(root + "miniworld/potsdam")
    os.makedirs(root + "miniworld/potsdam/train")
    os.makedirs(root + "miniworld/potsdam/test")

    os.system("/data/anaconda3/bin/python merge_mini_world.py ")
if whereIam == "wdtim719z":
    os.makedirs(root + "miniworld/toulouse")
    os.makedirs(root + "miniworld/toulouse/train")
    os.makedirs(root + "miniworld/toulouse/test")
    os.makedirs(root + "miniworld/potsdam")
    os.makedirs(root + "miniworld/potsdam/train")
    os.makedirs(root + "miniworld/potsdam/test")

    os.system("/data/anaconda3/bin/python merge_mini_world.py ")

print("miniworld has been successfully created")
