import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

# os.system("/d/jcastillo/anaconda3/bin/python -u train.py")

name = "borderless_size_2_early_97.csv"
os.system("/d/jcastillo/anaconda3/bin/python -u val.py")
os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
