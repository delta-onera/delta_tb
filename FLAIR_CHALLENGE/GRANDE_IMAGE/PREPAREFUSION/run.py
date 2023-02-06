print("ha non")
quit()


import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.system("mkdir build")

for name in ["RGB", "RIE", "IGE", "IEB"]:
    os.system("mkdir build/" + name)

    os.system("/d/achanhon/miniconda3/bin/python -u test.py " + name + " train")
    os.system("/d/achanhon/miniconda3/bin/python -u test.py " + name + " test")

    """
    l = os.listdir("build/" + name)
    moins, plus = 0, 0
    for i in l:
        x = torch.load("build/" + name + "/" + l)
        xmin, xmax = x.flatten().min(), x.flatten().max()
        if moins > xmin:
            moins = xmin
        if plus < xmax:
            plus = xmax
    minmax = torch.Tensor([moins, plus])
    torch.save(minmax, "build/minmax" + name + ".txt")
    """

os.system("rm -rf __pycache__")
