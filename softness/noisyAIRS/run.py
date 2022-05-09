import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

for noiselevel in range(3):
    os.system("/d/jcastillo/anaconda3/bin/python -u noisyairs.py " + noiselevel)

    configs = [
        ("0", "border"),  # more loss on border
        ("1", "border"),  # remove 1 border px + more loss on border
        ("0", "noborder"),  # standard
        ("1", "noborder"),  # remove border px
    ]
    for size, flag in configs:
        trainarg = "train.py 98 " + " " + size + " " + flag
        os.system("/d/jcastillo/anaconda3/bin/python -u " + trainarg)

        if flag == "noborder":
            trainname = "norm_size_" + size + "_"
        else:
            trainname = "bord_size_" + size + "_"
        for i in ["0", "1", "2"]:
            val = "VAL_" + train + i + ".csv"
            os.system("/d/jcastillo/anaconda3/bin/python -u val.py " + val + " " + i)

    os.system("rm -rf __pycache__")
