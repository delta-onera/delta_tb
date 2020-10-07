
import sys
print(sys.argv)

import numpy as np
import PIL
from PIL import Image
import os

nbclasses = 16
minnbpixel= 100
few = 3

def labelvector(im):
    out = np.zeros(nbclasses)
    for i in range(nbclasses):
        if np.sum((im==i).astype(int))>=minnbpixel:
            out[i]=1
    return out

def processtown(root):
    names = os.listdir(root)
    
    individuallabel = {}
    for name in names:
        tmp = PIL.Image.open(root+"/"+name).convert("L").copy()
        tmp =np.asarray(tmp,dtype=np.uint8)
        individuallabel[name] = labelvector(tmp)
    
    alllabel = np.ones(nbclasses)
    kept = []
    for i in range(few):
        tmp = [(np.sum(individuallabel[name]*alllabel),name) for name in names]
        tmp = sorted(tmp)
        add,name = tmp[-1]
        alllabel = alllabel-add
        kept = name
    
    alllabel = np.zeros(nbclasses)
    for name in kept:
        alllabel += individuallabel[name]
    
    print("using",kept,"allows to get classes", alllabel,"for town",root)
    
def processall(root)    
    names = os.listdir(root)
    for name in names:
        processtown(root+name)

processall("/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/UA")

