import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")

os.system("mkdir build")

# os.system("cp ../miniworld/build/model.pth build/")
# erosion 0
# Biarritz tensor([78.6980, 94.6378])
# Strasbourg tensor([71.9950, 90.6731])
# Paris tensor([75.2996, 96.8422])
# digitanie_toulouse tensor([72.7540, 92.2183])
# erosion 2
# Biarritz tensor([80.0219, 95.6642])
# Strasbourg tensor([74.5771, 92.5974])
# Paris tensor([75.4093, 97.2974])
# digitanie_toulouse tensor([76.6753, 94.3347])

os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")
# miniworld 443099 iter - 87% en train

os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")
