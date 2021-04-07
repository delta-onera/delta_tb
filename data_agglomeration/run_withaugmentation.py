import os
import sys

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

if os.path.exists("build"):
    os.system("rm -rf build")
os.makedirs("build")

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python withaugmentation.py")
    os.system("/data/anaconda3/envs/pytorch/bin/python test.py augmented.pth")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python withaugmentation.py")
    os.system("/d/jcastillo/anaconda3/bin/python test.py augmented.pth")
