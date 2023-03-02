import os
import sys

os.system("rm -rf build")
os.system("mkdir build")

for model in ["model1", "model2", "model3", "model4"]:
    os.system("mkdir build/" + model)
    os.system("/d/achanhon/miniconda3/bin/python -u test.py " + model)
