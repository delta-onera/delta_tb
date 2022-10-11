import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("/d/achanhon/miniconda3/bin/python train.py")
os.system("/d/achanhon/miniconda3/bin/python test.py")

os.system("rm -rf __pycache__")
