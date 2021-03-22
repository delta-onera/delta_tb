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

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratch_ai4geo/"

if os.path.exists(root + "miniworldtmp"):
    print("it seems miniworld exists, please remove it by hand")
    quit()
os.makedirs(root + "miniworldtmp")

if whereIam == "super":
    os.system("/data/anaconda3/envs/rahh/bin/python merge_mini_world.py ")

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/bin/python merge_mini_world.py ")

if whereIam == "ldtis706z":
    os.system("/data/anaconda3/bin/python merge_mini_world.py ")

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python merge_mini_world.py")


print("miniworld has been successfully created")
