

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
class MergedSegSemDataset:
    def __init__(self,alldatasets):
        self.alldatasets = alldatasets
        self.colorweights=[]
        
    def getrandomtiles(self,nbtiles,tilesize,batchsize):
        XY = []
        for dataset in self.alldatasets:
            xy = dataset.getrawrandomtiles((nbtiles//len(self.alldatasets))+1,tilesize)
            XY = x+y
        
        X = torch.stack([torch.Tensor(np.transpose(x,axes=(2, 0, 1))).cpu() for x,y in XY])
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x,y in XY])
        dataset = torch.utils.data.TensorDataset(X,Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)

        return dataloader
        
    def getCriterionWeight(self):
        return self.colorweights.copy()   

root = "/data/"
alldatasets = []

for i in range(1,len(sys.argv)):
    if sys.argv[i].find('*')>0:
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
    
nbclasses = 2
cm = np.zeros((nbclasses,nbclasses),dtype=int)

data = MergedSegSemDataset(alldatasets)
allfreq = np.zeros(2)
for singledataset in alldatasets:
    allfreq+=segsemdata.getBinaryFrequency(alllabels)
data.colorweights = [1.,1.*allfreq[0]/allfreq[1]]



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
