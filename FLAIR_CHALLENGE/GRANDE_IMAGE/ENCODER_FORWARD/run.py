import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.system("mkdir build")
os.system("mv " + sys.argv[1] + "/build/model.pth build/model.pth")

os.system("/d/achanhon/miniconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
