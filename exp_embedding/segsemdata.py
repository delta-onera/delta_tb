

import numpy as np

def safeuint8(x):
    x0 = np.zeros(x.shape,dtype=float)
    x255 = np.ones(x.shape,dtype=float)*255
    x = np.maximum(x0,np.minimum(x.copy(),x255))
    return np.uint8(x)

def symetrie(x,y,i,j,k):
    if i==1:
        x,y = np.transpose(x,axes=(1,0,2)),np.transpose(y,axes=(1,0))
    if j==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    if k==1:
        x,y = np.flip(x,axis=1),np.flip(y,axis=1)
    return x.copy(),y.copy()

def normalizehistogram(im):
    if len(im.shape)==2:
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

    else:
        output = im.copy()
        for i in range(im.shape[2]):
            output[:,:,i] = normalizehistogram(im[:,:,i])
        return output

import PIL
from PIL import Image

import torch
import torchvision

class SegSemDataset:
    def __init__(self,datasetname):
        #metadata
        self.datasetname = datasetname
        self.nbchannel = -1
        self.resolution = -1

        #vt structure
        self.setofcolors = []

        #path to data
        self.root = ""
        self.pathTOdata = {}

    def metadata(self):
        return (self.datasetname,self.nbchannel,len(self.setofcolors))

    def getnames(self):
        return [name for name in self.pathTOdata]

    def getImageAndLabel(self,name,innumpy=True):
        x,y = self.pathTOdata[name]

        if self.nbchannel==3:
            image = PIL.Image.open(self.root+"/"+x).convert("RGB").copy()
        else:
            image = PIL.Image.open(self.root+"/"+x).convert("L").copy()
        image = np.asarray(image,dtype=np.uint8) #warning wh swapping

        label = PIL.Image.open(self.root+"/"+y).convert("RGB").copy()
        label = self.colorvtTOvt(np.asarray(label,dtype=np.uint8)) #warning wh swapping

        if innumpy:
            return image, label
        else:
            if self.nbchannel == 3:
                image = torch.Tensor(np.transpose(image,axes=(2, 0, 1))).unsqueeze(0)
            else:    
                image = torch.Tensor(image).unsqueeze(0).unsqueeze(0)
            return image, label

    def getrawrandomtiles(self,nbtiles,tilesize):
        XY = []
        nbtilesperimage = nbtiles//len(self.pathTOdata)+1

        #crop
        for name in self.pathTOdata:
            image,label = self.getImageAndLabel(name)

            row = np.random.randint(0,image.shape[0]-tilesize-2,size = nbtilesperimage)
            col = np.random.randint(0,image.shape[1]-tilesize-2,size = nbtilesperimage)

            for i in range(nbtilesperimage):
                im = image[row[i]:row[i]+tilesize,col[i]:col[i]+tilesize,:].copy()
                mask = label[row[i]:row[i]+tilesize,col[i]:col[i]+tilesize].copy()
                XY.append((im,mask))

        #symetrie
        symetrieflag = np.random.randint(0,2,size = (len(XY),3))
        XY = [(symetrie(x,y,symetrieflag[i][0],symetrieflag[i][1],symetrieflag[i][2])) for i,(x,y) in enumerate(XY)]
        return XY

    def getrandomtiles(self,nbtiles,tilesize,batchsize):
        XY = self.getrawrandomtiles(nbtiles,tilesize)

        #pytorch
        if self.nbchannel == 3:
            X = torch.stack([torch.Tensor(np.transpose(x,axes=(2, 0, 1))).cpu() for x,y in XY])
        else:
            X = torch.stack([torch.Tensor(x).unsqueeze(0).cpu() for x,y in XY])
        Y = torch.stack([torch.from_numpy(y).long().cpu() for x,y in XY])
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
            mask1 = (maskcolor[:,:,0]==self.setofcolors[i][0]).astype(int)
            mask2 = (maskcolor[:,:,1]==self.setofcolors[i][1]).astype(int)
            mask3 = (maskcolor[:,:,2]==self.setofcolors[i][2]).astype(int)
            mask+=i*mask1*mask2*mask3

        return mask


    def copyTOcache(self,pathTOcache="build",outputresolution=-1, color=True, normalize=False, outputname=""):
        nativeresolution = self.resolution
        if outputresolution<0:
            outputresolution = nativeresolution
        if outputname=="":
            out = SegSemDataset(self.datasetname)
        else:
            out = SegSemDataset(outputname)

        if color:
            out.nbchannel = 3
        else:
            out.nbchannel = 1
        out.setofcolors = self.setofcolors.copy()
        out.resolution = outputresolution

        out.root = pathTOcache
        for name in self.pathTOdata:
            x,y = self.pathTOdata[name]

            if color:
                image = PIL.Image.open(self.root+"/"+x).convert("RGB").copy()
            else:
                image = PIL.Image.open(self.root+"/"+x).convert("L").copy()

            label = PIL.Image.open(self.root+"/"+y).convert("RGB").copy()

            if nativeresolution!=outputresolution:
                image = image.resize((int(image.size[0]*nativeresolution/outputresolution),int(image.size[1]*nativeresolution/outputresolution)), PIL.Image.BILINEAR)
                label = label.resize((image.size[0],image.size[1]), PIL.Image.NEAREST)

            label = out.vtTOcolorvt(out.colorvtTOvt(np.asarray(label,dtype=np.uint8))) #very slow but avoid frustrating bug due to label color coding
            label = PIL.Image.fromarray(label)

            if normalize:
                image = np.asarray(image,dtype=np.uint8)
                image = normalizehistogram(image)
                image = PIL.Image.fromarray(np.stack(image,axis=-1))

            image.save(out.root+"/"+name+"_x.png")
            label.save(out.root+"/"+name+"_y.png")
            out.pathTOdata[name] = (name+"_x.png",name+"_y.png")

        return out



def makeDFC2015(datasetpath="/data/DFC2015", lod0=True, dataflag="all"):
    dfc = SegSemDataset("DFC2015")
    dfc.nbchannel,dfc.resolution,dfc.root = 3,5,datasetpath

    if lod0:
        dfc.setofcolors = [[255,255,255],[0,0,255]]
    else:
        dfc.setofcolors = [[255,255,255]
            ,[0,0,128]
            ,[255,0,0]
            ,[0,255,255]
            ,[0,0,255]
            ,[0,255,0]
            ,[255,0,255]
            ,[255,255,0]]

    if dataflag not in ["all","fewshot","train","test"]:
        print("unknown flag in makeDFC2015",dataflag)
        quit()

    if dataflag == "test" or dataflag=="all":
        dfc.pathTOdata["5"]=("BE_ORTHO_27032011_315135_56865.tif","label_315135_56865.tif")
        dfc.pathTOdata["6"]=("BE_ORTHO_27032011_315145_56865.tif","label_315145_56865.tif")
    if dataflag == "train" or dataflag=="all":
        dfc.pathTOdata["1"]=("BE_ORTHO_27032011_315130_56865.tif","label_315130_56865.tif")
        dfc.pathTOdata["2"]=("BE_ORTHO_27032011_315130_56870.tif","label_315130_56870.tif")
        dfc.pathTOdata["3"]=("BE_ORTHO_27032011_315135_56870.tif","label_315135_56870.tif")
        dfc.pathTOdata["4"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    if dataflag == "fewshot":
        dfc.pathTOdata["4"]=("BE_ORTHO_27032011_315140_56865.tif","label_315140_56865.tif")
    

    return dfc

def makeISPRS(datasetpath="", lod0=True, dataflag="all", POTSDAM=True):
    if dataflag not in ["all","fewshot","train","test"]:
        print("unknown flag in makeISPRS",dataflag)
        quit()

    if POTSDAM:
        isprs = SegSemDataset("POTSDAM")
        isprs.nbchannel,isprs.resolution = 3,5
        if datasetpath=="":
            datasetpath = "/data/POSTDAM"
    else:
        isprs = SegSemDataset("VAIHINGEN")
        isprs.nbchannel,isprs.resolution = 3,10
        if datasetpath=="":
            datasetpath = "/data/VAIHINGEN"
    isprs.root = datasetpath

    if lod0:
        isprs.setofcolors = [[255,255,255],[0,0,255]]
    else:
        isprs.setofcolors = [[255, 255, 255]
            ,[0, 0, 255]
            ,[0, 255, 255]
            ,[ 0, 255, 0]
            ,[255, 255, 0]
            ,[255, 0, 0]]

    if POTSDAM:
        train = ["top_potsdam_2_10_",
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
            "top_potsdam_6_8_"]
        test = ["top_potsdam_6_9_",
            "top_potsdam_6_10_",
            "top_potsdam_6_11_",
            "top_potsdam_6_12_",
            "top_potsdam_7_7_",
            "top_potsdam_7_8_",
            "top_potsdam_7_9_",
            "top_potsdam_7_10_",
            "top_potsdam_7_11_",
            "top_potsdam_7_12_"]
        
        names = []
        if dataflag=="train":
            names = train
        if dataflag=="test":
            names = test
        if dataflag=="all":
            names = train+test
        if dataflag=="fewshot":
            names = ["top_potsdam_2_10_"]
        
        for name in names:
            isprs.pathTOdata[name] = ("2_Ortho_RGB/"+name+"RGB.tif","5_Labels_for_participants/"+name+"label.tif")

    else:
        train = ["top_mosaic_09cm_area5.tif",
            "top_mosaic_09cm_area17.tif",
            "top_mosaic_09cm_area21.tif",
            "top_mosaic_09cm_area23.tif",
            "top_mosaic_09cm_area26.tif",
            "top_mosaic_09cm_area28.tif",
            "top_mosaic_09cm_area30.tif",
            "top_mosaic_09cm_area32.tif",
            "top_mosaic_09cm_area34.tif",
            "top_mosaic_09cm_area37.tif"]
        test = ["top_mosaic_09cm_area1.tif",
            "top_mosaic_09cm_area3.tif",
            "top_mosaic_09cm_area7.tif",
            "top_mosaic_09cm_area11.tif",
            "top_mosaic_09cm_area13.tif",
            "top_mosaic_09cm_area15.tif"]

        names = []
        if dataflag=="train":
            names = train
        if dataflag=="test":
            names = test
        if dataflag=="all":
            names = train+test
        if dataflag=="fewshot":
            names = ["top_mosaic_09cm_area26"]

        for name in names:
            isprs.pathTOdata[name] = ("top/"+name,"gts_for_participants/"+name)

    return isprs


import os

def makeAIRSdataset(datasetpath="/data/AIRS/trainval", train=True):
    if train:
        allfile = os.listdir(datasetpath+"/train/image")
    else:
        allfile = os.listdir(datasetpath+"/val/image")

    airs = SegSemDataset("AIRS")
    airs.nbchannel,airs.resolution,airs.root,airs.setofcolors = 3,8,datasetpath,[[0,0,0],[255,255,255]]
    for name in allfile:
        if train:
            airs.pathTOdata[name] = ("/train/image/"+name,"/train/label/"+name[0:-4]+"_vis.tif")
        else:
            airs.pathTOdata[name] = ("/val/image/"+name,"/val/label/"+name[0:-4]+"_vis.tif")

    return airs

def makeINRIAdataset(datasetpath = "/data/INRIA/AerialImageDataset/train"):
    allfile = os.listdir(datasetpath+"/images")

    inria = SegSemDataset("INRIA")
    inria.nbchannel,airs.resolution,airs.root,airs.setofcolors = 3,50,datasetpath,[[0,0,0],[255,255,255]]
    for name in allfile:
        inria.pathTOdata[name] = ("images/"+name,"gt/"+name)

    return inria
