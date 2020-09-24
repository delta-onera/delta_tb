

import numpy as np

def safeuint8(x):
    x0 = np.zeros(x.shape,dtype=float)
    x255 = np.ones(x.shape,dtype=float)*255
    x = np.maximum(x0,np.minimum(x.copy(),x255))
    return np.uint8(x)
    
def symetrie(x,y,i,j,k):
    if i==1:
        x,y = np.transpose(x,axes=(1,0,2)),np.transpose(y,axes=(1,0,2))
    if j==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    if k==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    return x.copy(),y.copy()
        
def normalizehistogram(im):
    allvalues = list(im.flatten())
    allvalues = sorted(allvalues)
    n = len(allvalues)
    allvalues = allvalues[0:int(98*n/100)]
    allvalues = allvalues[int(2*n/100):]
    
    n = len(allvalues)
    k = n//255
    pivot = [0]+[allvalues[i] for i in range(0,n,k)]
    assert(len(pivot)>=255)
    
    out = np.zeros(im.shape,dtype = int)
    for i in range(1,255):
        out=np.maximum(out,np.uint8(im>pivot[i])*i)
        
    return np.uint8(out)


import PIL
from PIL import Image

class datasetdescription:
	def __init__(self):
		self.datasetname = ""
        self.nbclass = -1
        self.nbchannel = -1

class SegSemDataset:
    def __init__(self):
        self.datasetdescription = None
        
        #vt structure
        self.setofcolors = []
        
        #path to data
        self.root = ""
        self.names = []
        self.images = {}
        self.masks = {}
    
    def getdatasetdescription(self):
		return self.datasetdescription.copy()
		
    def getcolors(self):
        return self.setofcolors.copy()
    
    def getnames(self):
		return self.names.copy()
    
    def getImageMask(self,name):
		if self.nbchannel==3:
			image = PIL.Image.open(self.root+"/"+self.images[name]).convert("RGB").copy()
		if self.nbchannel==1:
			image = PIL.Image.open(self.root+"/"+self.images[name]).convert("L").copy()
		
		if self.nbchannel not in [1,3]:
			print("unacceptable channel size")
			quit()
        image = np.asarray(image,dtype=np.uint8) #warning wh swapping
        
        label = PIL.Image.open(self.root+"/"+self.masks[name]).convert("RGB").copy()
        label = colorvtTOvt(np.asarray(label,dtype=np.uint8)) #warning wh swapping
        
        return image, label
        
    
    def getrawrandomtiles(self,nbtilesperimage,h,w,batchsize):
        #crop
        XY = []
        for name in self.names:
            col = np.random.randint(0,self.sizes[name][0]-w-2,size = nbtilesperimage)
            row = np.random.randint(0,self.sizes[name][1]-h-2,size = nbtilesperimage)
                   
            image = PIL.Image.open(self.root+"/"+self.images[name]).convert("RGB").copy()
            image = np.asarray(image,dtype=np.uint8) #warning wh swapping

            label = PIL.Image.open(self.root+"/"+self.masks[name]).convert("RGB").copy()
            label = np.asarray(label,dtype=np.uint8) #warning wh swapping
            for i in range(nbtilesperimage):
                im = image[row[i]:row[i]+h,col[i]:col[i]+w,:].copy()
                mask = label[row[i]:row[i]+h,col[i]:col[i]+w,:].copy()
                XY.append((im,mask))
                        
        #symetrie
        symetrieflag = np.random.randint(0,2,size = (len(XY),3))
        XY = [(symetrie(x,y,symetrieflag[i][0],symetrieflag[i][1],symetrieflag[i][2])) for i,(x,y) in enumerate(XY)]
        return XY
        
    def getrandomtiles(self,nbtilesperimage,h,w,batchsize):
        XY = self.getrawrandomtiles(nbtilesperimage,h,w,batchsize)

        #pytorch
        X = torch.stack([torch.Tensor(np.transpose(x,axes=(2, 0, 1))).cpu() for x,y in XY])
        tmp = [torch.from_numpy(self.colorvtTOvt(y)).long().cpu() for x,y in XY]
        Y = torch.stack(tmp)
        dataset = torch.utils.data.TensorDataset(X,Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
        
        return dataloader
        
            
    def vtTOcolorvt(self,mask):
        maskcolor = np.zeros((mask.shape[0],mask.shape[1],3),dtype=int)
        for i in range(len(self.setofcolors)):
            for ch in range(3):
                maskcolor[:,:,ch]+=((mask == i).astype(int))*self.setofcolors[i][ch]
        return safeuint8(maskcolor)

    def colorvtTOvt(self,maskcolor):
        mask = np.zeros((maskcolor.shape[0],maskcolor.shape[1]),dtype=int)
        for i in range(len(self.setofcolors)):
            mask+=i*(maskcolor[:,:,0]==self.setofcolors[i][0]).astype(int)*(maskcolor[:,:,1]==self.setofcolors[i][1]).astype(int)*(maskcolor[:,:,2]==self.setofcolors[i][2]).astype(int)
        return mask


def resizeDataset(XY,path,setofcolors,nativeresolution,resolution,color,normalize,pathTMP="build"):
    XYout = {}
    for name,(x,y) in XY.items():
        image = PIL.Image.open(path+"/"+x).convert("RGB").copy()
        image = image.resize((int(image.size[0]*nativeresolution/resolution),int(image.size[1]*nativeresolution/resolution)), PIL.Image.BILINEAR)
        if not color:
            image = torchvision.transforms.functional.to_grayscale(image,num_output_channels=1)
        if normalize:
            image = np.asarray(image,dtype=np.uint8)
            image = normalizehistogram(image)
            image = PIL.Image.fromarray(np.stack([image,image,image],axis=-1))
        image.save(pathTMP+"/"+name+"_x.png")
        
        maskc = PIL.Image.open(path+"/"+y).convert("RGB").copy()
        maskc = maskc.resize((int(maskc.size[0]*nativeresolution/resolution),int(maskc.size[1]*nativeresolution/resolution)), PIL.Image.NEAREST)
        maskc.save(pathTMP+"/"+name+"_y.png")
        
        XYout[name] = (name+"_x.png",name+"_y.png")
    
    out = SegSemDataset()
    out.root = pathTMP
    out.setofcolors = setofcolors
    for name,(x,y) in XYout.items():
        out.names.append(name)
        out.sizes[name] = getsize(pathTMP+"/"+x)

        out.images[name] = x
        out.masks[name] = y
        
    return out
    


def DFC2015color(mode):
    if mode=="normal":
        return [[255,255,255]
        ,[0,0,128]
        ,[255,0,0]
        ,[0,255,255]
        ,[0,0,255]
        ,[0,255,0]
        ,[255,0,255]
        ,[255,255,0]]
    else:#lod0
        return [[255,255,255],[0,0,255]]


def makeDFC2015(resolution=50,datasetpath="/data/DFC2015",mode="lod0",color = False,normalize = True):
    XY = {}
    XY["1"]=("BE_ORTHO_27032011_315130_56865.tif","label_315130_56865.tif")
    XY["2"]=("BE_ORTHO_27032011_315130_56870.tif","label_315130_56870.tif")
    XY["3"]=("BE_ORTHO_27032011_315135_56870.tif","label_315135_56870.tif")
    XY["4"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    XY["5"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    XY["6"]=("BE_ORTHO_27032011_315145_56865.tif","label_315145_56865.tif")
    
    return resizeDataset(XY,datasetpath,DFC2015color(mode),5,resolution,color,normalize)    

def maketrainDFC2015(resolution=50,datasetpath="/data/DFC2015",mode="normal",color = True,normalize = False):
    XY = {}
    XY["1"]=("BE_ORTHO_27032011_315130_56865.tif","label_315130_56865.tif")
    XY["2"]=("BE_ORTHO_27032011_315130_56870.tif","label_315130_56870.tif")
    XY["3"]=("BE_ORTHO_27032011_315135_56870.tif","label_315135_56870.tif")
    XY["4"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    
    return resizeDataset(XY,datasetpath,DFC2015color(mode),5,resolution,color,normalize)
    
def maketestDFC2015(resolution=50,datasetpath="/data/DFC2015",mode="normal",color = True,normalize = False):
    XY = {}
    XY["5"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    XY["6"]=("BE_ORTHO_27032011_315145_56865.tif","label_315145_56865.tif")
    
    return resizeDataset(XY,datasetpath,DFC2015color(mode),5,resolution,color,normalize)

def ISPRScolor(mode):
    if mode=="normal":
        return [[255, 255, 255]
        ,[0, 0, 255]
        ,[0, 255, 255]
        ,[ 0, 255, 0]
        ,[255, 255, 0]
        ,[255, 0, 0]]
    else:#lod0
        return [[255,255,255],[0,0,255]]
       
def makeISPRS_VAIHINGEN(resolution=50,datasetpath="/data/ISPRS_VAIHINGEN",mode="lod0",color = False,normalize = True):
    names = ["top_mosaic_09cm_area1.tif",
    "top_mosaic_09cm_area3.tif",
    "top_mosaic_09cm_area5.tif",
    "top_mosaic_09cm_area7.tif",
    "top_mosaic_09cm_area11.tif",
    "top_mosaic_09cm_area13.tif",
    "top_mosaic_09cm_area15.tif",
    "top_mosaic_09cm_area17.tif",
    "top_mosaic_09cm_area21.tif",
    "top_mosaic_09cm_area23.tif",
    "top_mosaic_09cm_area26.tif",
    "top_mosaic_09cm_area28.tif",
    "top_mosaic_09cm_area30.tif",
    "top_mosaic_09cm_area32.tif",
    "top_mosaic_09cm_area34.tif",
    "top_mosaic_09cm_area37.tif"]
    
    XY = {}
    for name in names:
        XY[name]=("top/"+name,"gts_for_participants/"+name)
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),9,resolution,color,normalize)

def makeISPRStrainVAIHINGEN(resolution=50,datasetpath="/data/ISPRS_VAIHINGEN",mode="lod0",color = False,normalize = True):
    names = ["top_mosaic_09cm_area5.tif",
    "top_mosaic_09cm_area17.tif",
    "top_mosaic_09cm_area21.tif",
    "top_mosaic_09cm_area23.tif",
    "top_mosaic_09cm_area26.tif",
    "top_mosaic_09cm_area28.tif",
    "top_mosaic_09cm_area30.tif",
    "top_mosaic_09cm_area32.tif",
    "top_mosaic_09cm_area34.tif",
    "top_mosaic_09cm_area37.tif"]
    
    XY = {}
    for name in names:
        XY[name]=("top/"+name,"gts_for_participants/"+name)
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),9,resolution,color,normalize)
    
def makeISPRStestVAIHINGEN(resolution=50,datasetpath="/data/ISPRS_VAIHINGEN",mode="lod0",color = False,normalize = True):
    names = ["top_mosaic_09cm_area1.tif",
    "top_mosaic_09cm_area3.tif",
    "top_mosaic_09cm_area7.tif",
    "top_mosaic_09cm_area11.tif",
    "top_mosaic_09cm_area13.tif",
    "top_mosaic_09cm_area15.tif"]
    
    XY = {}
    for name in names:
        XY[name]=("top/"+name,"gts_for_participants/"+name)
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),9,resolution,color,normalize)
        

def makeISPRS_POSTDAM(resolution=50,datasetpath="/data/POSTDAM",mode="lod0",color = False,normalize = True):
    names = ["top_potsdam_2_10_",
    "top_potsdam_2_11_",
    "top_potsdam_2_12_",
    "top_potsdam_3_10_",
    "top_potsdam_3_11_",
    "top_potsdam_3_12_",
    "top_potsdam_4_10_",
    "top_potsdam_4_11_",
    "top_potsdam_4_12_",
    "top_potsdam_5_10_",
    "top_potsdam_5_11_",
    "top_potsdam_5_12_",
    "top_potsdam_6_7_",
    "top_potsdam_6_8_",
    "top_potsdam_6_9_",
    "top_potsdam_6_10_",
    "top_potsdam_6_11_",
    "top_potsdam_6_12_",
    "top_potsdam_7_7_",
    "top_potsdam_7_8_",
    "top_potsdam_7_9_",
    "top_potsdam_7_10_",
    "top_potsdam_7_11_",
    "top_potsdam_7_12_"]
    
    XY = {}
    for name in names:
        XY[name]=("2_Ortho_RGB/"+name+"RGB.tif","5_Labels_for_participants/"+name+"label.tif")
    
    return resizeDataset(XY,datasetpath,ISPRScolor(mode),5,resolution,color,normalize)




    
import os   
import random

def makeAIRSdataset(datasetpath,resolution=50,color = False,normalize = True,size=0):
    allfile = os.listdir(datasetpath+"/image")
    if size>0:
        random.shuffle(allfile)
        allfile = allfile[0:size]
    
    XY = {}
    for name in allfile:
        XY[name] = ("image/"+name,"label/"+name[0:-4]+"_vis.tif")
    return resizeDataset(XY,datasetpath, [[0,0,0],[255,255,255]],7.5,resolution,color,normalize)
    
def makeTrainAIRSdataset(datasetpath="/data/AIRS/train",resolution=50,color = False,normalize = True,size=33):
    allfile = os.listdir(datasetpath+"/image")
    if size>0:
        random.shuffle(allfile)
        allfile = allfile[0:size]
    
    XY = {}
    for name in allfile:
        XY[name] = ("image/"+name,"label/"+name[0:-4]+"_vis.tif")
    return resizeDataset(XY,datasetpath, [[0,0,0],[255,255,255]],7.5,resolution,color,normalize)    

def makeTestAIRSdataset(datasetpath="/data/AIRS/val",resolution=50,color = False,normalize = True,size=33):
    allfile = os.listdir(datasetpath+"/image")
    if size>0:
        random.shuffle(allfile)
        allfile = allfile[0:size]
    
    XY = {}
    for name in allfile:
        XY[name] = ("image/"+name,"label/"+name[0:-4]+"_vis.tif")
    return resizeDataset(XY,datasetpath, [[0,0,0],[255,255,255]],7.5,resolution,color,normalize)    

