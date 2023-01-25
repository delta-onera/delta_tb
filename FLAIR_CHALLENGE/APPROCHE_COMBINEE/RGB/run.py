import os
import sys

os.system("rm -rf build")
os.system("mkdir build")
os.system("cp -f ../dataloader.py build")
os.system("cp -f ../test.py build")
os.system("cp -f train.py build")
os.system("cd build")

os.system("/d/achanhon/miniconda3/bin/python -u train.py")
os.system("/d/achanhon/miniconda3/bin/python -u test.py")
