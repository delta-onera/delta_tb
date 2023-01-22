import os
import sys

os.system("rm -rf __pycache__")
os.system("mv build/model3ch.pth model3ch.pth")
os.system("rm -rf build")
os.makedirs("build")
os.system("mv model3ch.pth build")

os.system("/d/achanhon/miniconda3/bin/python -u train.py")
os.system("/d/achanhon/miniconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
