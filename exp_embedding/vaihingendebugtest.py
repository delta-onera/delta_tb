

print("TEST")
import torch
import segsemdata
import embedding
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("load data")
datatest = segsemdata.makeISPRS(datasetpath = "/data/ISPRS_VAIHINGEN",trainData=False,POSTDAM=False)
nbclasses = len(datatest.getcolors())
cm = np.zeros((nbclasses,nbclasses),dtype=int)
names=datatest.getnames()

with torch.no_grad():
	print("load model")
	net = torch.load("build/model.pth")
	net = net.to(device)
	net.eval()

	print("test")	
	for name in names:
		image,label = datatest.getImageAndLabel(name,innumpy=False)
		pred = net(image.to(device)).cpu()
		
		
		cm+= confusion_matrix(allproducts[i][0].flatten(),allproducts[i][1].flatten(),list(range(nbclasses)))
	print("accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
	print(cm)

