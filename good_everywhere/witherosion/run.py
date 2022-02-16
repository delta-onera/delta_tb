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
# erosion 0
# Biarritz tensor([79.4944, 94.6580])
# Strasbourg tensor([79.3251, 94.0217])
# Paris tensor([72.1782, 96.1099])
# digitanie_toulouse tensor([71.7173, 91.8314])
# erosion 2
# Biarritz tensor([85.2892, 96.7749])
# Strasbourg tensor([79.9626, 94.5913])
# Paris tensor([72.6626, 96.5655])
# digitanie_toulouse tensor([74.3848, 93.2833])

# miniworld converged
# erosion 2
# Biarritz tensor([84.5727, 96.4907, 96.1224, 73.0231])
# Strasbourg tensor([79.0587, 94.3362, 93.6887, 64.4287])
# Paris tensor([74.1588, 96.9264, 96.8227, 51.4949])
# digitanie_toulouse tensor([70.7350, 91.9572, 91.2469, 50.2230])
# erosion 0
# Biarritz tensor([77.8856, 93.9718, 93.2995, 62.4716])
# Strasbourg tensor([78.3292, 93.6979, 92.9126, 63.7459])
# Paris tensor([73.5063, 96.4684, 96.3354, 50.6772])
# digitanie_toulouse tensor([67.8489, 90.1676, 89.2510, 46.4468])
# bref c'est nul !!!


os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")