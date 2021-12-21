import os
import sys

whereIam = os.uname()[1]
assert whereIam in ["wdtim719z", "calculon", "astroboy", "flexo", "bender", "ldtis706z"]

if whereIam == "wdtim719z":
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratchf/"

if not os.path.exists(root + "miniworld"):
    print("miniworld not found")
    quit()

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

if whereIam == "wdtim719z":
    os.system("/data/anaconda3/envs/pytorch/bin/python -u train.py baseline")
    os.system("/data/anaconda3/envs/pytorch/bin/python -u test.py | tee logbasline.txt")
    os.system("/data/anaconda3/envs/pytorch/bin/python -u train.py penalizemin")
    os.system("/data/anaconda3/envs/pytorch/bin/python -u test.py | tee logpenalty.txt")
if whereIam == "ldtis706z":
    os.system("python3 -u train.py baseline")
    os.system("python3 -u test.py | tee logbasline.txt")
    os.system("python3 -u train.py penalizemin")
    os.system("python3 -u test.py | tee logpenalty.txt")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python -u train.py baseline")
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py | tee logbasline.txt")
    os.system("/d/jcastillo/anaconda3/bin/python -u train.py penalizemin")
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py | tee logpenalty.txt")


os.system("rm -rf __pycache__")
