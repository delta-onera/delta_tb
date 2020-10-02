

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

print("load model")
net = embedding.Embedding(pretrained="/data/vgg16-00b39a1b.pth")
net = net.to(device)

print("load data")
datatrain = segsemdata.makeISPRS(datasetpath = "/data/ISPRS_VAIHINGEN",POTSDAM=False)
datatrain = datatrain.copyTOcache(outputresolution=50)
net.adddataset(datatrain.metadata())
net = net.to(device)
nbclasses = len(datatrain.setofcolors)
earlystopping = datatrain.getrandomtiles(1000,128,16)

print("train setting")
import torch.nn as nn

import collections
import random
from sklearn.metrics import confusion_matrix
criterion = nn.CrossEntropyLoss()
optimizer = net.getoptimizer()

meanloss = collections.deque(maxlen=200)
nbepoch = 90

def trainaccuracy():
    net.eval()
    cm = np.zeros((nbclasses,nbclasses),dtype=int)
    with torch.no_grad():
        for inputs, targets in earlystopping:
            inputs = inputs.to(device)
            outputs = net(inputs,datatrain.metadata())
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
        
        preds = net(inputs,datatrain.metadata())
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
