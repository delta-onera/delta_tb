
import sys
print(sys.argv)

import numpy as np
import PIL
from PIL import Image
import os

nbclasses = 16
realclasse = [0,1,4,6,9,10,13]

def labelvector(im):
    out = np.zeros(nbclasses)
    for i in range(nbclasses):
        out[i]=np.sum((im==i).astype(int))
        
    outafter = np.zeros(nbclasses)
    for i in realclasse:
        outafter[i]=np.sum((im==i).astype(int))
    for i in range(nbclasses):
        if i not in realclasse:
            outafter[15]+=np.sum((im==i).astype(int))
    
    return out.astype(int),outafter.astype(int)

def processtown(root):
    #print(root)
    names = os.listdir(root)
    
    individuallabel = {}
    for name in names:
        tmp = PIL.Image.open(root+"/"+name).convert("L").copy()
        tmp =np.asarray(tmp,dtype=np.uint8)
        tmp = labelvector(tmp)
        individuallabel[name]=tmp
    
    alllabel,alllabelafter = np.zeros(nbclasses).astype(int),np.zeros(nbclasses).astype(int)
    for name in individuallabel:
        a,b = individuallabel[name]
        alllabel+=a
        alllabelafter+=b
    
    print((100.*alllabel/np.sum(alllabel)).astype(int),(100.*alllabelafter/np.sum(alllabelafter)).astype(int),root) 
    return alllabel,alllabelafter
   
def processall(root):
    names = ["Angers","Caen", "Cherbourg", "Lille_Arras_Lens_Douai_Henin",
            "Marseille_Martigues", "Nice", "Rennes" ,"Vannes","Brest",
            "Calais_Dunkerque", "Clermont-Ferrand", "LeMans" ,"Lorient",
            "Nantes_Saint-Nazaire", "Quimper", "Saint-Brieuc"]

    alllabel,alllabelafter = np.zeros(nbclasses).astype(int),np.zeros(nbclasses).astype(int)
    for name in names:
        a,b = processtown(root+"/"+name)
        alllabel+=a
        alllabelafter+=b
    
    #print(alllabel) 
    print((100.*alllabel/np.sum(alllabel)).astype(int),(100.*alllabelafter/np.sum(alllabelafter)).astype(int))

processall("/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/UA")

