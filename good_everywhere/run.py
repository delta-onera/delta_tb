import os
import sys

whereIam = os.uname()[1]
assert whereIam in ["wdtim719z", "calculon", "astroboy", "flexo", "bender", "ldtis706z"]

if whereIam == "wdtim719z":
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratch_ai4geo/"

if not os.path.exists(root + "miniworld"):
    print("miniworld not found")
    quit()

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

if whereIam == "wdtim719z":
    cities = [
        "potsdam",
        "christchurch",
        "toulouse",
        "austin",
        "chicago",
        "kitsap",
        "tyrol-w",
        "vienna",
        "bruges",
        "Arlington",
        "Austin",
        "DC",
        "NewYork",
        "SanFrancisco",
        "Atlanta",
        "NewHaven",
        "Norfolk",
        "Seekonk",
    ]
    for city in cities:
        print(city)
        os.system("/data/anaconda3/envs/pytorch/bin/python -u normal.py " + city)
        os.system("/data/anaconda3/envs/pytorch/bin/python -u test.py" + city)
if whereIam == "ldtis706z":
    print("freq")
    os.system("python3 -u freq.py")
    os.system("python3 -u test.py debug")
    print("normal")
    os.system("python3 -u normal.py")
    os.system("python3 -u test.py debug")
    print("min")
    os.system("python3 -u min.py")
    os.system("python3 -u test.py debug")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    print("freq")
    os.system("/d/jcastillo/anaconda3/bin/python -u freq.py")
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py")
    # print("normal")
    # os.system("/d/jcastillo/anaconda3/bin/python -u normal.py")
    # os.system("/d/jcastillo/anaconda3/bin/python -u test.py")
    # print("min")
    # os.system("/d/jcastillo/anaconda3/bin/python -u min.py")
    # os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
