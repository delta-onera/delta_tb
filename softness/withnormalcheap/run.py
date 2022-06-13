import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

python = "/d/jcastillo/anaconda3/bin/python -u "

os.system(python + "noisyairs.py nonoise 100")
os.system(python + "train.py")
os.system(python + "val.py")

os.system("rm -rf __pycache__")
