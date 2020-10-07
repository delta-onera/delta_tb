
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
    return out.astype(int)

def processtown(root):
    #print(root)
    names = os.listdir(root)
    
    individuallabel = {}
    for name in names:
        tmp = PIL.Image.open(root+"/"+name).convert("L").copy()
        tmp =np.asarray(tmp,dtype=np.uint8)
        tmp = labelvector(tmp)
        if np.sum(tmp)!=0:
            individuallabel[name]=tmp
    
    alllabel = np.zeros(nbclasses)
    for name in individuallabel:
        alllabel+=individuallabel[name]
    alllabel = (alllabel>0).astype(int)
    
    alllabelcopy = alllabel.copy()
    kept = []
    for i in range(few):
        alllabel = alllabelcopy.copy()
        for name in kept:
            alllabel-=individuallabel[name]
        alllabel = (alllabel>0).astype(int)    
        #print(i,alllabel)
        
        tmp = [(np.sum(individuallabel[name]*alllabel),name) for name in individuallabel if name not in kept]
        tmp = sorted(tmp)
        add,name = tmp[-1]
        alllabel = alllabel-add
        kept.append(name)
    
    alllabel = np.zeros(nbclasses)
    for name in kept:
        alllabel += individuallabel[name]
    alllabel = (alllabel>0).astype(int)
    
    #print("town",root,"contains classes",alllabelcopy)
    #print("using",kept,"allows to get classes", alllabel)
    #if np.sum(np.abs(alllabelcopy-alllabel))==0:
    #    print("ok")
    #for name in kept:
    #    print(individuallabel[name])
    print(root,alllabelcopy,kept) 
   
def processall(root):
    names = ["Angers","Caen", "Cherbourg", "Lille_Arras_Lens_Douai_Henin",
            "Marseille_Martigues", "Nice", "Rennes" ,"Vannes","Brest",
            "Calais_Dunkerque", "Clermont-Ferrand", "LeMans" ,"Lorient",
            "Nantes_Saint-Nazaire", "Quimper", "Saint-Brieuc"]

    for name in names:
        processtown(root+"/"+name)

processall("/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/UA")

