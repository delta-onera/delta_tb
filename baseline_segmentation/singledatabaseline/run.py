import os
import sys

if len(sys.argv) == 1:
    print("YOU NEED TO PROVIDE PATH TO DATA")
    quit()

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("python3 -u train.py" + sys.argv)
os.system("python3 -u test.py" + sys.argv)

os.system("rm -rf __pycache__")
