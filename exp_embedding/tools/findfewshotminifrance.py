
import sys
print(sys.argv)

import numpy as np
import PIL
from PIL import Image
import os

nbclasses = 16
realclasse = [1,2,5,7,10,11,14]
minnbpixel= 100
few = 3

def labelvector(im):
    out = np.zeros(nbclasses)
    for i in realclasse:
        if np.sum((im==i).astype(int))>=minnbpixel:
            out[i]=1
    return out

def processtown(root):
    print(root)
    names = os.listdir(root)
    
    individuallabel = {}
    for name in names:
        tmp = PIL.Image.open(root+"/"+name).convert("L").copy()
        tmp =np.asarray(tmp,dtype=np.uint8)
        individuallabel[name] = labelvector(tmp)
    
    alllabel = np.zeros(nbclasses)
    for name in names:
        alllabel+=individuallabel[name]
    alllabel = (alllabel>0).astype(int)
    
    alllabelcopy = alllabel.copy()
    kept = []
    for i in range(few):
        alllabel = np.zeros(nbclasses)
        for name in names:
            alllabel+=individuallabel[name]
        alllabel = (alllabel>0).astype(int)
        
        for name in kept:
            alllabel-=individuallabel[name]
        alllabel = (alllabel>0).astype(int)    
        print(i,alllabel)
        
        tmp = [(np.sum(individuallabel[name]*alllabel),name) for name in names]
        tmp = sorted(tmp)
        add,name = tmp[-1]
        alllabel = alllabel-add
        kept.append(name)
    
    alllabel = np.zeros(nbclasses)
    for name in kept:
        alllabel += individuallabel[name]
    alllabel = (alllabel>0).astype(int)
    
    print("town",root,"contains classes",alllabelcopy)
    print("using",kept,"allows to get classes", alllabel)
    for name in kept:
        print(individuallabel[name])
    
def processall(root):
    names = ["Angers","Caen", "Cherbourg", "Lille_Arras_Lens_Douai_Henin",
            "Marseille_Martigues", "Nice", "Rennes" ,"Vannes","Brest",
            "Calais_Dunkerque", "Clermont-Ferrand", "LeMans" ,"Lorient",
            "Nantes_Saint-Nazaire", "Quimper", "Saint-Brieuc"]

    for name in names:
        processtown(root+"/"+name)

processall("/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/UA")

