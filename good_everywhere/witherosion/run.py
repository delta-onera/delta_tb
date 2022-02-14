import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")

os.system("mkdir build")
# os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")
# miniworld 30000 iter - 85% en train
# erosion 2
# Biarritz tensor([84.5982, 96.4833])
# Strasbourg tensor([80.4862, 94.6770])
# Paris tensor([73.6460, 96.3646])
# digitanie_toulouse tensor([73.3204, 92.8653])

# os.system("cp ../miniworld/build/model.pth build/")
# erosion 0
# Biarritz tensor([79.1583, 94.7050])
# Strasbourg tensor([69.5644, 88.9260])
# Paris tensor([72.7991, 96.3264])
# digitanie_toulouse tensor([71.3615, 91.8434])
# erosion 2
# Biarritz tensor([80.6257, 95.9263])
# Strasbourg tensor([74.4799, 93.2062])
# Paris tensor([73.4836, 97.1685])
# digitanie_toulouse tensor([76.5187, 94.6876])

os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")
# miniworld 443099 iter - 87% en train


if len(sys.argv) == 1:
    os.system("/d/achanhon/miniconda/bin/python test.py")
else:
    if len(sys.argv) == 2 and sys.argv[1] == "nohup":
        os.system("nohup /d/achanhon/miniconda/bin/python test.py &")
    else:
        os.system(sys.argv[1] + " test.py")


os.system("rm -rf __pycache__")
