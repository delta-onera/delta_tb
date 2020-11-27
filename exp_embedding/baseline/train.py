

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

sys.path.append('../..')
import segsemdata



root = "/data/"
if len(sys.argv)==1 or (datasetname not in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE"]):
    datasetname = "VAIHINGEN"
else:
    datasetname = sys.argv[1]
print("load data",datasetname)

if datasetname == "VAIHINGEN":
    datatest = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN",dataflag="train",POTSDAM=False)
if datasetname == "POSTDAM":
    datatest = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM",dataflag="train",POTSDAM=True)
if datasetname == "BRUGES":
    datatest = segsemdata.makeDFC2015(datasetpath = root+"DFC2015",dataflag="train")
if datasetname == "TOULOUSE":
    print("TODO DATALOADER")
    quit()

datatest = datatest.copyTOcache(outputresolution=50)
nbclasses = len(datatest.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatest.getnames()



#### TODO conditional import depending on the model
namemodel = "UNET"
if (len(sys.argv)==3 and sys.argv[2]=="UNET") or True:
    import unet
    print("load model",namemodel)

    net = unet.UNET(nbclasses)
    net = net.to(device,pretrained = "/data/vgg16-00b39a1b.pth")



print("train setting")
import torch.nn as nn
import collections
import random
from sklearn.metrics import confusion_matrix

criterion = nn.CrossEntropyLoss()
optimizer = net.getoptimizer()
meanloss = collections.deque(maxlen=200)
nbepoch = 90

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
