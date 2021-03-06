

import sys
print(sys.argv)
assert(len(sys.argv)>1)

import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

sys.path.append('../..')
import segsemdata



print("load data")
root = "/data/"
assert(sys.argv[1] in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE","VAIHINGEN_lod0","POTSDAM_lod0","BRUGES_lod0","TOULOUSE_lod0","AIRS"])
if sys.argv[1] == "VAIHINGEN":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN",dataflag="train",POTSDAM=False)
if sys.argv[1] == "VAIHINGEN_lod0":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN", labelflag="lod0",weightflag="iou",dataflag="train",POTSDAM=False)
if sys.argv[1] == "POTSDAM":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM",dataflag="train",POTSDAM=True)
if sys.argv[1] == "POTSDAM_lod0":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM", labelflag="lod0",weightflag="iou",dataflag="train",POTSDAM=True)
if sys.argv[1] == "BRUGES":
    data = segsemdata.makeDFC2015(datasetpath = root+"DFC2015",dataflag="train")
if sys.argv[1] == "BRUGES_lod0":
    data = segsemdata.makeDFC2015(datasetpath = root+"DFC2015", labelflag="lod0",weightflag="iou",dataflag="train")
if sys.argv[1] == "TOULOUSE":
    data = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="train")
if sys.argv[1] == "TOULOUSE_lod0":
    data = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="train", labelflag="lod0",weightflag="iou")  
if sys.argv[1] == "AIRS":
    data = segsemdata.makeAIRSdataset(datasetpath = root+"AIRS",dataflag="train",weightflag="iou")  
  
if sys.argv[1] in ["TOULOUSE","TOULOUSE_lod0"] or len(sys.argv)==2 or sys.argv[2] not in ["grey","normalize"]:
    data = data.copyTOcache(outputresolution=50)
else:
    if sys.argv[2]=="grey":
        data = data.copyTOcache(outputresolution=50,color=False)
    else:
        data = data.copyTOcache(outputresolution=50,color=False,normalize=True)
nbclasses = len(data.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)



print("load unet")
import unet
net = unet.UNET(nbclasses,pretrained="/data/vgg16-00b39a1b.pth")
net = net.to(device)



print("train setting")
import torch.nn as nn
import collections
import torch.optim as optim
import random
from sklearn.metrics import confusion_matrix

weigths = torch.Tensor(data.getCriterionWeight()).to(device)
criterion = nn.CrossEntropyLoss(weight=weigths)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 120

earlystopping = data.getrandomtiles(1000,128,16)
def trainaccuracy():
    net.eval()
    cm = np.zeros((nbclasses,nbclasses),dtype=int)
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _,pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(pred[i].cpu().numpy().flatten(),targets[i].cpu().numpy().flatten(),list(range(nbclasses)))
    return segsemdata.getstat(cm)



print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    trainloader = data.getrandomtiles(2000,128,16)
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        preds = net(inputs)
        loss = criterion(preds,targets)
        meanloss.append(loss.cpu().data.numpy())

        if epoch>30:
            loss = loss*0.5
        if epoch>60:
            loss = loss*0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if random.randint(0,30)==0:
            print("loss=",(sum(meanloss)/len(meanloss)))

    torch.save(net, "build/model.pth")
    acc,iou,IoU=trainaccuracy()
    print("stat:", acc,iou,IoU)
    if acc>0.97:
        quit()
