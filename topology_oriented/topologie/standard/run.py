import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("python3 train.py")
os.system("python3 test.py")

os.system("rm -rf __pycache__")
