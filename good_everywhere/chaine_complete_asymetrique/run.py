import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("/d/jcastillo/anaconda3/bin/python -u train.py")
os.system("/d/jcastillo/anaconda3/bin/python -u val.py")
os.system("/d/jcastillo/anaconda3/bin/python -u test.py 2")

os.system("rm -rf __pycache__")
