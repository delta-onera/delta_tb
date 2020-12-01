

import sys
print(sys.argv)
assert(len(sys.argv)>1)
assert(sys.argv[1] in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE","VAIHINGEN_lod0","POTSDAM_lod0","BRUGES_lod0","TOULOUSE_lod0"])

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



print("load data")
root = "/data/"
if sys.argv[1] == "VAIHINGEN":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN",dataflag="test",POTSDAM=False)
if sys.argv[1] == "VAIHINGEN_lod0":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN", labelflag="lod0",weightflag="iou",dataflag="test",POTSDAM=False)
if sys.argv[1] == "POTSDAM":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM",dataflag="test",POTSDAM=True)
if sys.argv[1] == "POTSDAM_lod0":
    data = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM", labelflag="lod0",weightflag="iou",dataflag="test",POTSDAM=True)
if sys.argv[1] == "BRUGES":
    data = segsemdata.makeDFC2015(datasetpath = root+"DFC2015",dataflag="test")
if sys.argv[1] == "BRUGES_lod0":
    data = segsemdata.makeDFC2015(datasetpath = root+"DFC2015", labelflag="lod0",weightflag="iou",dataflag="test")
if sys.argv[1] == "TOULOUSE":
    data = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="test")
if sys.argv[1] == "TOULOUSE_lod0":
    data = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="test", labelflag="lod0",weightflag="iou")  
  
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
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)



print("test")
with torch.no_grad():
    net.eval()
    for name in data.getnames():
        image,label = data.getImageAndLabel(name,innumpy=False)
        pred = net(image.to(device))
        _,pred = torch.max(pred[0],0)
        pred = pred.cpu().numpy()

        assert(label.shape==pred.shape)

        cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))

        pred = PIL.Image.fromarray(data.vtTOcolorvt(pred))
        pred.save("build/"+name+"_z.png")

    print(getstat(cm))
    print(cm)

