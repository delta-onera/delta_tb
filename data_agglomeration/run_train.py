import os
import sys

whereIam = os.uname()[1]
assert whereIam in ["super", "wdtis719z", "ldtis706z"]

if whereIam in ["super", "wdtis719z"]:
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"

if not os.path.exists(root + "miniworld"):
    print("run merge before")
    quit()

if os.path.exists("build"):
    os.rmdir("build")
os.makedirs("build")

if whereIam == "super":
    os.system("/data/anaconda3/bin/python train.py ")

if whereIam == "ldtis706z":
    os.system("/data/anaconda3/bin/python train.py ")
