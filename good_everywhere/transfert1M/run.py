import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")
os.system("mkdir build")

os.system("cp ../segmentation1M/build/model.pth build/")

if len(sys.argv) == 1:
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py 0")
else:
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py " + sys.argv[1])

os.system("rm -rf __pycache__")


# 119999 perf tensor([85.6451, 96.4592])
# 5
# Biarritz tensor([83.9187, 96.5573, 96.2300, 71.6074])
# Strasbourg tensor([78.0971, 93.2429, 92.3264, 63.8678])
# Paris tensor([73.9376, 97.0816, 96.9906, 50.8846])
# digitanie_toulouse tensor([70.9204, 91.5611, 90.7451, 51.0958])
