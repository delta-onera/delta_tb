import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("/d/jcastillo/anaconda3/bin/python -u noisyairs.py 1")

configs = [
    ("0", "97", "noborder"),
    ("0", "95", "border"),
]
for size, stop, flag in configs:
    trainarg = "train.py " + stop + " " + size + " " + flag
    os.system("/d/jcastillo/anaconda3/bin/python -u " + trainarg)

    if flag == "noborder":
        trainname = "bas_size_1_stop_95_"
    else:
        trainname = "foc_size_1_stop_95_"
    for i in ["0", "1", "2"]:
        val = "VAL_" + train + "size_" + i + ".csv"
        os.system("/d/jcastillo/anaconda3/bin/python -u val.py " + val + " " + i)

os.system("rm -rf __pycache__")
