import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")
os.system("mkdir build")

os.system("cp ../learning_1M/build/model.pth build/")

if len(sys.argv) == 1:
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py 2")
else:
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py " + sys.argv[1])

os.system("rm -rf __pycache__")
