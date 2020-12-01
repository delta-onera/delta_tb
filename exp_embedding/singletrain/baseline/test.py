

import sys
print(sys.argv[0])
assert(len(sys.argv)==1)
assert(sys.argv[1] in ["VAIHINGEN","POTSDAM","BRUGES","TOULOUSE"])



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
    datatest = segsemdata.makeISPRS(datasetpath = root+"ISPRS_VAIHINGEN",dataflag="test",POTSDAM=False)
if datasetname == "POTSDAM":
    datatest = segsemdata.makeISPRS(datasetpath = root+"ISPRS_POTSDAM",dataflag="test",POTSDAM=True)
if datasetname == "BRUGES":
    datatest = segsemdata.makeDFC2015(datasetpath = root+"DFC2015",dataflag="test")
if datasetname == "TOULOUSE":
    datatest = segsemdata.makeSEMCITY(datasetpath = root+"SEMCITY_TOULOUSE",dataflag="test")

datatest = datatest.copyTOcache(outputresolution=50)
nbclasses = len(datatest.setofcolors)
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatest.getnames()



#### TODO conditional import depending on the model
namemodel = "UNET"
if (len(sys.argv)==3 and sys.argv[2]=="UNET") or True:
    import unet

    print("load model",namemodel)
    with torch.no_grad():
        net = torch.load("build/model.pth")
        net = net.to(device)



print("test")
with torch.no_grad():
    net.eval()
    for name in names:
        image,label = datatest.getImageAndLabel(name,innumpy=False)
        pred = net(image.to(device))
        _,pred = torch.max(pred[0],0)
        pred = pred.cpu().numpy()

        assert(label.shape==pred.shape)

        cm+= confusion_matrix(label.flatten(),pred.flatten(),list(range(nbclasses)))

        pred = PIL.Image.fromarray(datatest.vtTOcolorvt(pred))
        pred.save("build/"+name+"_z.png")

    print("accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
    print(cm)

