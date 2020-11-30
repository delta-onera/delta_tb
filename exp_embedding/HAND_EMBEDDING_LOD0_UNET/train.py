

import sys
print(sys.argv[0])

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

sys.path.append('..')
import segsemdata



root = "/data/"
if len(sys.argv)==1 or (sys.argv[1] not in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE"]):
    datasetname = "VAIHINGEN"
else:
    datasetname = sys.argv[1]
print("load data",datasetname)

if datasetname == "VAIHINGEN":
    datatrain = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN",dataflag="train",POTSDAM=False)
if datasetname == "POTSDAM":
    datatrain = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM",dataflag="train",POTSDAM=True)
if datasetname == "BRUGES":
    datatrain = segsemdata.makeDFC2015(datasetpath = root+"DFC2015",dataflag="train")
if datasetname == "TOULOUSE":
    datatrain = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="train")

datatrain = datatrain.copyTOcache(outputresolution=50)
nbclasses = len(datatrain.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatrain.getnames()



#### TODO conditional import depending on the model
namemodel = "UNET"
if (len(sys.argv)==3 and sys.argv[2]=="UNET") or True:
    import unet
    print("load model",namemodel)

    net = unet.UNET(nbclasses,pretrained="/data/vgg16-00b39a1b.pth")
    net = net.to(device)



print("train setting")
import torch.nn as nn
import collections
import torch.optim as optim
import random
from sklearn.metrics import confusion_matrix

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
meanloss = collections.deque(maxlen=200)
nbepoch = 120

earlystopping = datatrain.getrandomtiles(1000,128,16)
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
    return np.sum(cm.diagonal())/(np.sum(cm)+1)



print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    trainloader = datatrain.getrandomtiles(2000,128,16)
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
    acc=trainaccuracy()
    print("acc=", acc)
    if acc>0.97:
        quit()
