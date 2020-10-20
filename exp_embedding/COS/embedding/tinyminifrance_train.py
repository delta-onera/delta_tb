

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

print("load model")
net = embedding.Embedding(pretrained="/home/achanhon/vgg16-00b39a1b.pth")
net = net.to(device)

print("load data")
datatrain = {}
nbclasses = {} 
earlystopping = {}
for town in knowntown:
    tmp = segsemdata.makeTinyMiniFrancePerTown(datasetpath = "/data01/PUBLIC_DATASETS/MiniFrance/tmFrance/",town=town,dataflag="train")
    datatrain[town] = tmp
    nbclasses[town] = len(tmp.setofcolors)
    earlystopping[town] = tmp.getrandomtiles(1000,256,16)
    net.adddataset(tmp.metadata())

net = net.to(device)

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
    cm = {}
    for town in knowntown:
        cm[town] = np.zeros((nbclasses[town],nbclasses[town]),dtype=int)
        
    with torch.no_grad():
        for town in knowntown:
            for inputs, targets in earlystopping[town]:
                inputs = inputs.to(device)
                outputs = net(inputs,datatrain[town].metadata())
                _,pred = outputs.max(1)
                for i in range(pred.shape[0]):
                    cm += confusion_matrix(pred[i].cpu().numpy().flatten(),targets[i].cpu().numpy().flatten(),list(range(nbclasses[town])))
    for town in knowntown:
        cm[town] = cm[town][1:,1:]
    return [np.sum(cm[town].diagonal())/(np.sum(cm[town])+1) for town in knowntown]

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    net.train()
    
    trainloader = {}
    for town in knowntown:
        trainloader[town] = datatrain.getrandomtiles(2000,256,16)
    
    iterators = {}
    for town in knowntown:
        iterators[town] = iter(trainloader[town])
    
    copytown = knowntown.copy()
    while True:
        if copytown.empty():
            break
        
        i = random.randint(0,len(copytown)-1)
        town = copytown[i]
        
        inputs, targets = next(iterators[town],(None,None))
        
        if inputs is None:
            copytown = [left for left in copytown if left!=town]
            continue
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        preds = net(inputs,datatrain[town].metadata())
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
    if all([a>0.97 for a in acc]):
        quit()
