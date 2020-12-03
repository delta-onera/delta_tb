

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
alldatasets = []

for i in range(1,len(sys.argv)):
    if sys.argv[i][-1]=='*':
        mode = "all"
        name = sys.argv[i][:-1]
    else:
        mode = "train"
        name = sys.argv[i]
    assert(name in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE","AIRS"])

    if name == "VAIHINGEN":
        data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN", labelflag="lod0",weightflag="iou",dataflag=mode,POTSDAM=False)
    if name == "POTSDAM":
        data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM", labelflag="lod0",weightflag="iou",dataflag=mode,POTSDAM=True)
    if name == "BRUGES":
        data = segsemdata.makeDFC2015(datasetpath = root+"DFC2015", labelflag="lod0",weightflag="iou",dataflag=mode)
    if name == "TOULOUSE":
        data = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag=mode, labelflag="lod0",weightflag="iou") 
    if name == "AIRS":
        data = segsemdata.makeAIRSdataset(datasetpath = root+"AIRS",dataflag=mode,weightflag="iou")  
  
    alldatasets.append(data.copyTOcache(outputresolution=50,color=False,normalize=True))



print("load embedding")
import embedding
net = embedding.Embedding(pretrained="/data/vgg16-00b39a1b.pth")
for data in alldatasets:
    net.adddataset(data.metadata())
net = net.to(device)



print("train setting")
import torch.nn as nn
import collections
import torch.optim as optim
import random
from sklearn.metrics import confusion_matrix

optimizer = net.getoptimizer()
weights = {}
criterion = {}
earlystopping = {}
nbclasses = {}
for data in alldatasets:
    weights[data.datasetname] = torch.Tensor(data.getCriterionWeight()).to(device)
    criterion[data.datasetname] = nn.CrossEntropyLoss(weight=weights[data.datasetname])
    earlystopping[data.datasetname] = data.getrandomtiles(1000,128,16)
    nbclasses[data.datasetname] = len(data.setofcolors)
    
meanloss = collections.deque(maxlen=200)
nbepoch = 120

def trainaccuracy(data):
    nbclasses_=nbclasses[data.datasetname]
    cm = np.zeros((nbclasses_,nbclasses_),dtype=int)
    with torch.no_grad():
        for inputs, targets in earlystopping[data.datasetname]:
            inputs = inputs.to(device)
            outputs = net(inputs,data.metadata())
            _,pred = outputs.max(1)
            for i in range(pred.shape[0]):
                cm += confusion_matrix(pred[i].cpu().numpy().flatten(),targets[i].cpu().numpy().flatten(),list(range(nbclasses_)))
    return segsemdata.getstat(cm)

def trainaccuracyall():
    net.eval()
    ACC = 0
    for data in alldatasets: 
        acc,iou,IOU = trainaccuracy(data)
        ACC+=acc
    return ACC/len(alldatasets)



print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    net.train()
    
    trainloader = {}
    iterators = {}
    for data in alldatasets: 
        trainloader[data.datasetname] = data.getrandomtiles(2000,128,16)
        iterators[data.datasetname] = iter(trainloader[data.datasetname])
    
    assert(2000//16*16==2000)
        
    for iteration in range(2000//16):
        optimizer.zero_grad()
        
        ###batch accumulation over dataset
        losses = []
        for data in alldatasets: 
            inputs, targets = next(iterators[data.datasetname])
            inputs,targets = inputs.to(device),targets.to(device)
            preds = net(inputs,data.metadata())
            tmp = criterion[data.datasetname](preds,targets)
            tmp.backward(retain_graph=True)
            losses.append(tmp.cpu().data.numpy())
        loss = sum(losses)
        ###batch accumulation over dataset    
        
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
    acc=trainaccuracyall()
    print("average acc:", acc)
    if acc>0.97:
        quit()
