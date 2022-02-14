import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")

os.system("mkdir build")

os.system("cp ../miniworld/build/model.pth build/")

# os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")
# miniworld 443099 iter - 87% en train

os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
