import os
import sys

os.system("rm -rf build")
os.system("mkdir build")

os.system("/d/achanhon/miniconda3/bin/python -u train.py")
os.system("/d/achanhon/miniconda3/bin/python -u ../test.py RGB/build")
