import os
import sys

# os.system("rm -rf __pycache__")
# os.system("rm -rf build")
# os.makedirs("build")

# os.system("/d/jcastillo/anaconda3/bin/python -u train.py")

train = "base_size_2_stop_97_"
for i in ["0", "1", "2"]:
    val = "VAL_" + train + "size_" + i + ".csv"
    os.system("/d/jcastillo/anaconda3/bin/python -u val.py " + val + " " + i)

    test = "TEST_" + train + "size_" + i + ".csv"
    os.system("/d/jcastillo/anaconda3/bin/python -u test.py" + test + " " + i)

os.system("rm -rf __pycache__")
