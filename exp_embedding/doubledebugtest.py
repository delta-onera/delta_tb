

import sys
print(sys.argv[0])

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

print("load data")
datatest = segsemdata.makeISPRS(datasetpath = "/data/ISPRS_VAIHINGEN",trainData=False,POTSDAM=False)
datatest = datatest.copyTOcache(outputresolution=70)
nbclasses = len(datatest.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatest.getnames()

with torch.no_grad():
    print("load model")
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()

    print("test")    
    for name in names:
        image,label = datatest.getImageAndLabel(name,innumpy=False)
        pred = net(image.to(device),datatest.metadata())
        _,pred = torch.max(pred[0],0)
        pred = pred.cpu().numpy()
        
        assert(label.shape==pred.shape)
        
        cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))
        
        pred = PIL.Image.fromarray(datatest.vtTOcolorvt(pred))
        pred.save("build/"+name+"_z.jpg")
        
    print("accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
    print(cm)
    
print("load data")
datatest = segsemdata.makeDFC2015(trainData=False)
datatest = datatest.copyTOcache(outputresolution=70)
nbclasses = len(datatest.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatest.getnames()

with torch.no_grad():
    print("load model")
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()

    print("test")    
    for name in names:
        image,label = datatest.getImageAndLabel(name,innumpy=False)
        pred = net(image.to(device),datatest.metadata())
        _,pred = torch.max(pred[0],0)
        pred = pred.cpu().numpy()
        
        if label.shape!=pred.shape:
            print(label.shape,pred.shape)
            quit()
        
        cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))
        
        pred = PIL.Image.fromarray(datatest.vtTOcolorvt(pred))
        pred.save("build/"+name+"_z.jpg")
        
    print("accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
    print(cm)

