

import sys
print(sys.argv)
sys.path.append('../..')

knowntown = ["Angers","Caen","Cherbourg","Lille_Arras_Lens_Douai_Henin",
    "Marseille_Martigues","Nice","Rennes","Vannes","Brest","Calais_Dunkerque",
    "Clermont-Ferrand","LeMans","Lorient","Nantes_Saint-Nazaire","Quimper",
    "Saint-Brieuc"]

import torch
import torch.backends.cudnn as cudnn
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    
import segsemdata
import embedding
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

with torch.no_grad():
    print("load model")
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()

    for town in knowntown:
        print("load data",town)
        datatest = segsemdata.makeTinyMiniFrancePerTown(datasetpath = "/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/",town=town,dataflag="test")
        nbclasses = len(datatest.setofcolors)
        cm = np.zeros((nbclasses,nbclasses),dtype=int)
        names=datatest.getnames()


        print("test")    
        for name in names:
            image,label = datatest.getImageAndLabel(name,innumpy=False)
            pred = net(image.to(device),datatest.metadata(),tilesize=256)
            _,pred = torch.max(pred[0],0)
            pred = pred.cpu().numpy()
            
            assert(label.shape==pred.shape)
            
            cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))
            
            pred = PIL.Image.fromarray(datatest.vtTOcolorvt(pred)*17)
            pred.save("build/"+name+"_z.png")
            pred = PIL.Image.fromarray(datatest.vtTOcolorvt(label)*17)
            pred.save("build/"+name+"_y.png")
            pred = PIL.Image.fromarray(np.transpose(image[0].cpu().numpy(),axes=(1,2,0)).astype(np.uint8))
            pred.save("build/"+name+"_x.jpg")
        
        cm = cm[1:,1:]    
        print("------------------------ accuracy",town,np.sum(cm.diagonal())/(np.sum(cm)+1))
        print(cm)

