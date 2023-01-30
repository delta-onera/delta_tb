import os
import sys

os.system("rm -rf build")
os.system("mkdir build")
os.system("cp -r ../FUSION/build/model.pth build")

os.system("/d/achanhon/miniconda3/bin/python -u test.py")
