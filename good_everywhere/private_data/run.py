import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")

os.system("mkdir build")
os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")

if len(sys.argv) == 1:
    os.system("/d/achanhon/miniconda/bin/python test.py")
else:
    os.system(sys.argv[1])  # for nohup + CUDA_VISIBLE_DEVICES=xxx

os.system("rm -rf __pycache__")
