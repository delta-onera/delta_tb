import os
import sys

whereIam = os.uname()[1]
assert whereIam in ["super", "wdtis719z", "ldtis706z"]

if whereIam in ["super", "wdtis719z"]:
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"

if os.path.exists(root + "miniworld"):
    print(
        "it seems miniworld exists, please remove it by hand to be sure you are ok to do that"
    )
    quit()

os.makedirs(root + "miniworld")
if whereIam == "super":    
    os.makedirs(root + "miniworld/toulouse")
    os.makedirs(root + "miniworld/toulouse/train")
    os.makedirs(root + "miniworld/toulouse/test")
    os.makedirs(root + "miniworld/potsdam")
    os.makedirs(root + "miniworld/potsdam/train")
    os.makedirs(root + "miniworld/potsdam/test")

    os.system("/data/anaconda3/envs/rahh/bin/python building_mini_world.py ")
