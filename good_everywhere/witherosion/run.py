import os
import sys

os.system("rm -rf build")
os.system("rm -rf __pycache__")

os.system("mkdir build")
os.system("cp ../../baseline_segmentation/baseline8bits/build/model.pth build/")
# erosion 2
# Biarritz tensor([84.5982, 96.4833])
# Strasbourg tensor([80.4862, 94.6770])
# Paris tensor([73.6460, 96.3646])
# digitanie_toulouse tensor([73.3204, 92.8653])


if len(sys.argv) == 1:
    os.system("/d/achanhon/miniconda/bin/python test.py")
else:
    if len(sys.argv) == 2 and sys.argv[1] == "nohup":
        os.system("nohup /d/achanhon/miniconda/bin/python test.py &")
    else:
        os.system(sys.argv[1] + " test.py")


os.system("rm -rf __pycache__")
