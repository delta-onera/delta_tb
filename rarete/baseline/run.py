import os

os.system("rm -rf __pycache__")
os.system("rm -r build")
os.system("mkdir build")

os.system("/d/achanhon/miniconda3/python -u train.py")
os.system("/d/achanhon/miniconda3/python3 -u test.py")

os.system("rm -rf __pycache__")
