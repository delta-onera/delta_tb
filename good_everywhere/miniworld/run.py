import os
import sys

os.system("rm -rf __pycache__")
os.system("rm -rf build")
os.makedirs("build")

os.system("/d/jcastillo/anaconda3/bin/python -u train.py")
os.system("/d/jcastillo/anaconda3/bin/python -u test.py")

os.system("rm -rf __pycache__")


# appris sur potsdam+christchurch+bradburry
# => 90% sur potsdam+90% sur christchurch !

# appris sur potsdam+christchurch+bradburry+pologne+rio ?
# =>
