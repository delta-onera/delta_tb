import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

for noise in ["nonoise", "hallucination", "pm1image", "pm1translation"]:
    for resolution in ["50cm", "1m"]:
        name = "AIRS_" + noise + "_" + resolution + "_"
        os.system("/d/jcastillo/anaconda3/bin/python -u noisyairs.py " + noiselevel)

        for method in ["base", "base+bord", "base-bord"]:
            os.system("/d/jcastillo/anaconda3/bin/python train.py " + method)

            name = name + method + "_"
            os.system("/d/jcastillo/anaconda3/bin/python -u val.py " + name)

    os.system("rm -rf __pycache__")
