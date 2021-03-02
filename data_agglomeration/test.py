import sys
import os
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

whereIam = os.uname()[1]
print(whereIam,sys.argv)
assert(whereIam in ["super","wdtis719z","ldtis706z"])


print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()

import dataloader

print("massif benchmark")
cm = {}

with torch.no_grad():
    if whereIam in ["super","wdtis719z"]:
        availabledata=["toulouse","potsdam"]
        root = "/data/miniworld/"
        
    if whereIam in ["ldtis706z"]:
        availabledata=["toulouse","potsdam","bruges","newzealand"]
        root = "/media/achanhon/bigdata/data/miniworld/"
    
    for name in availabledata:
        print("load",name)
        data = dataloader.SegSemDataset(root+name+"/test",name in ["toulouse","potsdam","bruges","newzealand"])

		cm[name] = np.zeros((2,2), dtype=int)
		for i in range(data.nbImages):
			image, label = data.getImageAndLabel(i, innumpy=False)
			image = image.to(device)
			
			globalresize = nn.AdaptiveAvgPool2d((image.shape[2],image.shape[3]))
            power2resize = nn.AdaptiveAvgPool2d((data.shape[2]//32)*32,(data.shape[3]//32)*32)
			
			pred = (net, image.to(device))
			_, pred = torch.max(pred[0], 0)
			pred = pred.cpu().numpy()

			assert label.shape == pred.shape

			cm[name] += confusion_matrix(
				label.flatten(), pred.flatten(), list(range(nbclasses))
			)

			pred = PIL.Image.fromarray(data.vtTOcolorvt(pred))
			pred.save("build/" + name +"_"+ str( + "_z.png")

		print(data.datasetname)
		print(segsemdata.getstat(cm))
		print(cm)
