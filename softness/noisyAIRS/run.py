import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

python = "/d/jcastillo/anaconda3/bin/python -u "
for noise in ["nonoise", "hallucination", "pm1image", "pm1translation"]:
    for resolution in ["50cm", "1m"]:
        name = "AIRS_" + noise + "_" + resolution + "_"
        os.system(python + "noisyairs.py " + noise + " " + resolution)

        for method in ["base", "base+bord", "base-bord"]:
            os.system(python + "train.py " + method)

            name = name + method + "_"
            os.system(python + "val.py " + name)

    os.system("rm -rf __pycache__")
