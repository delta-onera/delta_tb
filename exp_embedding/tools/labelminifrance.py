
import sys
print(sys.argv)

import numpy as np
import PIL
from PIL import Image
import os

nbclasses = 16

def labelvector(im):
    out = np.zeros(nbclasses)
    for i in realclasse:
        out[i]=np.sum((im==i).astype(int))
    return out.astype(int)

def processtown(root):
    #print(root)
    names = os.listdir(root)
    
    individuallabel = {}
    for name in names:
        tmp = PIL.Image.open(root+"/"+name).convert("L").copy()
        tmp =np.asarray(tmp,dtype=np.uint8)
        tmp = labelvector(tmp)
        individuallabel[name]=tmp
    
    alllabel = np.zeros(nbclasses)
    for name in individuallabel:
        alllabel+=individuallabel[name]
    
    print(root)
    print(alllabel) 
    print(1.*alllabel/np.sum(alllabel)) 
    return alllabel
   
def processall(root):
    names = ["Angers","Caen", "Cherbourg", "Lille_Arras_Lens_Douai_Henin",
            "Marseille_Martigues", "Nice", "Rennes" ,"Vannes","Brest",
            "Calais_Dunkerque", "Clermont-Ferrand", "LeMans" ,"Lorient",
            "Nantes_Saint-Nazaire", "Quimper", "Saint-Brieuc"]

    alllabel = np.zeros(nbclasses)
    for name in names:
        alllabel+=processtown(root+"/"+name)
    
    print(alllabel) 
    print(1.*alllabel/np.sum(alllabel)) 
    return alllabel

processall("/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/UA")

