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

if os.path.exists(root + "miniworld"):
    print("it seems miniworld exists, please remove it by hand")
    quit()
os.makedirs(root + "miniworld")

if whereIam == "super":
    os.system("/data/anaconda3/envs/rahh/bin/python merge_mini_world.py ")

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/bin/python merge_mini_world.py ")

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python merge_mini_world.py")


print("miniworld has been successfully created")

print(
    "WARNING -- there is an issue with norfolk building file -- remove the test file, put train/1_*.png instead"
)
