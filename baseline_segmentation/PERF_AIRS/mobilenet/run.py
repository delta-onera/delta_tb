import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("python3 -u train.py")
os.system("python3 -u test.py")

os.system("rm -rf __pycache__")
