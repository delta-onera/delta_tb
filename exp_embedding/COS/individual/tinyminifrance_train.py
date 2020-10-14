

import sys
print(sys.argv)
sys.path.append('../..')

knowntown = ["Angers","Caen","Cherbourg","Lille_Arras_Lens_Douai_Henin",
    "Marseille_Martigues","Nice","Rennes","Vannes","Brest","Calais_Dunkerque",
    "Clermont-Ferrand","LeMans","Lorient","Nantes_Saint-Nazaire","Quimper",
    "Saint-Brieuc"]
if len(sys.argv)==1 or sys.argv[1] not in knowntown:
    town = "Nice"
else:
    town = sys.argv[1]

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
net = embedding.Embedding(pretrained="/home/achanhon/vgg16-00b39a1b.pth")
net = net.to(device)

print("load data")
datatrain = segsemdata.makeTinyMiniFrancePerTown(datasetpath = "/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/",town=town,dataflag="train")
net.adddataset(datatrain.metadata())
net = net.to(device)
nbclasses = len(datatrain.setofcolors)
earlystopping = datatrain.getrandomtiles(2000,128,16)

print("train setting")
import torch.nn as nn

import collections
import random
from sklearn.metrics import confusion_matrix
weights = torch.tensor([0.]+[1.]*(nbclasses-1)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
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
    cm = cm[1:,1:]
    return np.sum(cm.diagonal())/(np.sum(cm)+1)

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    trainloader = datatrain.getrandomtiles(4000,128,16)
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
