import os

os.system("rm -rf build")
os.system("mkdir build")
os.system("cp -r ../../cheat/orientedfusion/build/predictions ./build/input")
os.system("mkdir build/output")

os.system("/d/achanhon/miniconda3/bin/python -u train.py")
os.system("/d/achanhon/miniconda3/bin/python -u test.py")
