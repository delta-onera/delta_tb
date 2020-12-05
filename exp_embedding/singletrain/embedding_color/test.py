

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
        mode = "test"
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

    alldatasets.append(data.copyTOcache(outputresolution=50))



print("load embedding")
import embedding
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)



print("test")
with torch.no_grad():
    net.eval()
    for data in alldatasets:
        nbclasses = 2
        cm = np.zeros((nbclasses,nbclasses),dtype=int)
        for name in data.getnames():
            image,label = data.getImageAndLabel(name,innumpy=False)
            pred = net(image.to(device),data.metadata())
            _,pred = torch.max(pred[0],0)
            pred = pred.cpu().numpy()

            assert(label.shape==pred.shape)

            cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))

            pred = PIL.Image.fromarray(data.vtTOcolorvt(pred))
            pred.save("build/"+name+"_z.png")

        print(data.datasetname)
        print(segsemdata.getstat(cm))
        print(cm)
