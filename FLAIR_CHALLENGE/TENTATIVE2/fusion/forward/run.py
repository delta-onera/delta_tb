import os
import sys

os.system("rm -rf build")
os.system("mkdir build")
os.system("cp ../baseline/build/model.pth build/model.pth")

os.system("/d/achanhon/miniconda3/bin/python -u test.py")
