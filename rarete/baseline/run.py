import os

os.system("rm -rf __pycache__")
os.system("rm -r build")
os.system("mkdir build")

os.system("python3 -u train.py")
os.system("python3 -u test.py")

os.system("rm -rf __pycache__")
