import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.system("mkdir build")

for name in ["RGB", "RIE", "IGE", "IEB"]:
    os.system("mkdir build/" + name)
    os.system("mkdir build/" + name + "/train")
    os.system("mkdir build/" + name + "/test")

    os.system("/d/achanhon/miniconda3/bin/python -u test.py " + name + " train")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py " + name + " test")

os.system("rm -rf __pycache__")
