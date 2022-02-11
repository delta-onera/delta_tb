import os
import sys
import time

if len(sys.argv) == 1:
    print("YOU NEED TO PROVIDE PATH TO DATA")
    quit()

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

t = time.time()
os.system("python3 -u train.py " + sys.argv[1] + "/train/")
os.system("python3 -u test.py " + sys.argv[1] + "/test/")
print("total duration", time.time() - t)

os.system("rm -rf __pycache__")
