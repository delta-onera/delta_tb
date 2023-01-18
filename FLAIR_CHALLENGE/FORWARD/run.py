import os
import sys

os.system("rm -rf __pycache__")
os.system("mv build/model.pth model.pth")
os.system("rm -rf build")
os.makedirs("build")
os.system("mv model.pth build")

os.system("/d/achanhon/miniconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
