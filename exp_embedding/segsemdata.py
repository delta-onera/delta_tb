

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

def makeMiniFrancePerTown(datasetpath="/data/minifrance",town="Nice",dataflag="all"):
    if dataflag not in ["all","fewshot","train","test"]:
        print("unknown flag in makeMiniFrancePerTown",dataflag)
        quit()

    knowntown = ["Angers","Caen","Cherbourg","Lille_Arras_Lens_Douai_Henin",
        "Marseille_Martigues","Nice","Rennes","Vannes","Brest","Calais_Dunkerque",
        "Clermont-Ferrand","LeMans","Lorient","Nantes_Saint-Nazaire","Quimper",
        "Saint-Brieuc"]
    if town not in knowntown:
        print("unknown town in makeMiniFrancePerTown",town)
        quit()
    
    minifrance = SegSemDataset("MiniFrance_"+town)
    minifrance.nbchannel,minifrance.resolution = 3,50
    minifrance.root = datasetpath
    minifrance.setofcolors = [[i,i,i] for i in [0,1,2,5,7,10,11,14]]
    minifrance.town = town
    
    if dataflag=="all":
        allfile = os.listdir(datasetpath+"/UA/"+town)
        for name in allfile:
            minifrance.pathTOdata[name] = ("BDORTHO/"+town+"/"+name,"UA/"+town+"/"+name)
    
    fewshot = {
        "Angers" : ['49-2013-0440-6705-LA93-0M50-E080_339_2354.tif', '49-2013-0450-6710-LA93-0M50-E080_3176_8597.tif', '49-2013-0415-6705-LA93-0M50-E080_976_3529.tif'],
        "Caen" : ['14-2012-0470-6910-LA93-0M50-E080_3340_4814.tif', '14-2012-0470-6910-LA93-0M50-E080_8563_1409.tif', '14-2012-0470-6910-LA93-0M50-E080_826_2069.tif'],
        "Cherbourg" : ['50-2012-0375-6955-LA93-0M50-E080_2474_371.tif', '50-2012-0375-6950-LA93-0M50-E080_5570_4215.tif', '50-2012-0380-6960-LA93-0M50-E080_8312_2519.tif'],
        "Lille_Arras_Lens_Douai_Henin" : ['59-2012-0675-7070-LA93-0M50-E080_5467_2098.tif', '62-2012-0700-7045-LA93-0M50-E080_8716_7689.tif', '62-2012-0710-7020-LA93-0M50-E080_6584_8644.tif'],
        "Marseille_Martigues" : ['13-2014-0925-6290-LA93-0M50-E080_8889_5451.tif', '13-2014-0890-6295-LA93-0M50-E080_6271_5801.tif', '13-2014-0925-6295-LA93-0M50-E080_8229_8797.tif'],
        "Nice" : ['06-2014-1045-6320-LA93-0M50-E080_6969_7103.tif', '06-2014-1050-6315-LA93-0M50-E080_4393_6503.tif', '06-2014-1045-6340-LA93-0M50-E080_7644_2359.tif'],
        "Rennes" : ['35-2012-0335-6780-LA93-0M50-E080_122_7029.tif', '35-2012-0370-6800-LA93-0M50-E080_7968_1161.tif', '35-2012-0370-6800-LA93-0M50-E080_7399_3835.tif'],
        "Vannes" : ['56-2013-0260-6760-LA93-0M50-E080_4803_8542.tif', '56-2013-0285-6750-LA93-0M50-E080_7324_8776.tif', '56-2013-0285-6750-LA93-0M50-E080_4910_2657.tif'],
        "Brest" : ['29-2012-0155-6850-LA93-0M50-E080_6977_7498.tif', '29-2012-0170-6830-LA93-0M50-E080_3404_8749.tif', '29-2012-0170-6830-LA93-0M50-E080_6071_214.tif'],
        "Calais_Dunkerque" : ['62-2012-0610-7085-LA93-0M50-E080_6945_8076.tif', '62-2012-0625-7095-LA93-0M50-E080_3164_4671.tif', '62-2012-0600-7065-LA93-0M50-E080_2898_2040.tif'],
        "Clermont-Ferrand" : ['63-2013-0710-6510-LA93-0M50-E080_4035_3983.tif', '63-2013-0735-6510-LA93-0M50-E080_8795_6491.tif', '63-2013-0735-6510-LA93-0M50-E080_1116_8958.tif'],
        "LeMans" : ['72-2013-0505-6765-LA93-0M50-E080_5108_3755.tif', '72-2013-0505-6765-LA93-0M50-E080_58_8678.tif', '72-2013-0500-6785-LA93-0M50-E080_824_2209.tif'],
        "Lorient" : ['56-2013-0235-6775-LA93-0M50-E080_3651_2565.tif', '56-2013-0230-6760-LA93-0M50-E080_8959_7032.tif', '56-2013-0240-6775-LA93-0M50-E080_3776_2484.tif'],
        "Nantes_Saint-Nazaire" : ['44-2013-0375-6685-LA93-0M50-E080_2774_4627.tif', '44-2013-0375-6705-LA93-0M50-E080_4847_2827.tif', '44-2013-0375-6705-LA93-0M50-E080_2298_8926.tif'],
        "Quimper" : ['29-2012-0185-6790-LA93-0M50-E080_7521_1692.tif', '29-2012-0185-6795-LA93-0M50-E080_8340_307.tif', '29-2012-0185-6795-LA93-0M50-E080_7135_60.tif'],
        "Saint-Brieuc" : ['22-2012-0280-6830-LA93-0M50-E080_8687_2233.tif', '22-2012-0285-6840-LA93-0M50-E080_7779_5785.tif', '22-2012-0285-6840-LA93-0M50-E080_612_8355.tif']
    }
            
    if dataflag=="fewshot":
        for name in fewshot[town]:
            minifrance.pathTOdata[name] = ("BDORTHO/"+town+"/"+name,"UA/"+town+"/"+name)
    
    if dataflag=="train":
        allfile = os.listdir(datasetpath+"/UA/"+town)
        allfile = sorted(allfile)
        allfile = allfile[0:int(60*len(allfile)/100)]
        allfile = list(set(allfile+fewshot[town]))
        
        for name in allfile:
            minifrance.pathTOdata[name] = ("BDORTHO/"+town+"/"+name,"UA/"+town+"/"+name)
            
    if dataflag=="test":
        allfile = os.listdir(datasetpath+"/UA/"+town)
        allfile = sorted(allfile)
        train = allfile[0:int(60*len(allfile)/100)]
        train = set(train+fewshot[town])
        
        allfile = [name for name in allfile if name not in train]
        
        for name in allfile:
            minifrance.pathTOdata[name] = ("BDORTHO/"+town+"/"+name,"UA/"+town+"/"+name)
    
    return minifrance
    
