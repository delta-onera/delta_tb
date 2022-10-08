import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

python = "python3"
os.system("python train.py")
os.system("python test.py")

os.system("rm -rf __pycache__")
