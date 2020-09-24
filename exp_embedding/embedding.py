
import torch
import torch.nn as nn
import nn.functional as F
import torch.optim as optim

class Encoding(nn.Module):
    def __init__(self,nbchannel):
        super(Encoding, self).__init__()
        
        self.conv11 = nn.Conv2d(nbchannel, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
    def forward(self, x):    
        return F.leaky_relu(self.conv12(F.leaky_relu(self.conv11(x))))

        
class Head(nn.Module):
    def __init__(self,nbclasses):
        super(Head, self).__init__()
        
        self.final1 = nn.Conv2d(224, 128, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(128, nbclasses, kernel_size=3, padding=1)
        
    def forward(self, x):    
        return self.final2(F.leaky_relu(self.final1(x)))


class UnetBackbone(nn.Module):
    def __init__(self,nbclasses):
        super(UnetBackbone, self).__init__()

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.compress5 = nn.Conv2d(512, 32, kernel_size=1)

        self.conv43d = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.compress4 = nn.Conv2d(256, 32, kernel_size=1)

        self.conv33d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.compress3 = nn.Conv2d(128, 32, kernel_size=1)
        
        self.conv22d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, x1):
        globalresize = nn.AdaptiveAvgPool2d((x.shape[2],x.shape[3]))
        
        tmp = F.max_pool2d(x1, kernel_size=2, stride=2)
        
        x2 = F.leaky_relu(self.conv21(tmp))
        x2 = F.leaky_relu(self.conv22(x2))
        tmp = F.max_pool2d(x2, kernel_size=2, stride=2)

        x3 = F.leaky_relu(self.conv31(tmp))
        x3 = F.leaky_relu(self.conv32(x3))
        x3 = F.leaky_relu(self.conv33(x3))
        tmp = F.max_pool2d(x3, kernel_size=2, stride=2)

        x4 = F.leaky_relu(self.conv41(tmp))
        x4 = F.leaky_relu(self.conv42(x4))
        x4 = F.leaky_relu(self.conv43(x4))
        tmp = F.max_pool2d(x4, kernel_size=2, stride=2)

        x5 = F.leaky_relu(self.conv51(tmp))
        x5 = F.leaky_relu(self.conv52(x5))
        x5 = F.leaky_relu(self.conv53(x5))
        
        tmp = F.upsample_nearest(x5, scale_factor=2)
        x5 = self.compress5(x5)
        x5 = globalresize(x5)

        x4 = torch.cat((tmp, x4), 1)
        x4 = F.leaky_relu(self.conv43d(x4))
        x4 = F.leaky_relu(self.conv42d(x4))
        x4 = F.leaky_relu(self.conv41d(x4))
        
        tmp = F.upsample_nearest(x4, scale_factor=2)
        x4 = self.compress4(x4)
        x4 = globalresize(x4)

        x3 = torch.cat((tmp, x3), 1)
        x3 = F.leaky_relu(self.conv33d(x3))
        x3 = F.leaky_relu(self.conv32d(x3))
        x3 = F.leaky_relu(self.conv31d(x3))
        
        tmp = F.upsample_nearest(x3, scale_factor=2)
        x3 = self.compress3(x3)
        x3 = globalresize(x3)
        
        x2 = torch.cat((tmp, x2), 1)
        x2 = F.leaky_relu(self.conv22d(x2))
        x2 = F.leaky_relu(self.conv21d(x2))
        x2 = globalresize(x2)

        x1 = torch.cat((x1, x2, x3, x4, x5), 1)        
        return x1
        
        
class Embedding(nn.Module):
    def __init__(self,nbclasses):
        super(Embedding, self).__init__()
        
        self.backbone = UnetBackbone()
        self.datahash = set()
        self.dictionnary = nn.ParameterDict()

    def simpleforward(self, data, datahash):
        return self.dictionnary[datahash+"_head"](self.backbone(self.dictionnary[datahash+"_encoder"](data)))

    def getoptimizer(flag="all"):
        if flag=="all":
            return optim.Adam(net.parameters(), lr=0.0001)
        
        if flag in datasetdescription.hash:
            return optim.Adam(list(self.dictionnary[flag+"_head"].parameters())+list(self.dictionnary[flag+"_encoder"].parameters()), lr=0.0001)
    
        print("unknown flag in getoptimizer")
        quit()


    def forward(self, data, datasetdescription):
        if datasetdescription.hash not in self.datahash:
            tmp = nn.ParameterDict({
                datasetdescription.hash+"_encoder": Encoder(datasetdescription.nbchannel),
                datasetdescription.hash+"_head": Head(datasetdescription.nbclasses)
            })
            self.dictionnary.update(tmp)
            self.datahash.insert(datasetdescription.hash)
         
        if 128 <= data.shape[2] <= 512 and data.shape[2]%32==0 and 128 <= data.shape[3] <= 512 and data.shape[3]%32==0:
            return self.simpleforward(x,datasetdescription.hash)
        
        if data.shape[2] <= 512 and data.shape[3] <= 512:
            globalresize = nn.AdaptiveAvgPool2d((data.shape[2],data.shape[3]))
            power2resize = nn.AdaptiveAvgPool2d((min(128,(data.shape[2]//32)*32),min(128,(data.shape[3]//32)*32)))
            
            data = power2resize(data)
            data = self.simpleforward(data,datasetdescription.hash)
            data = globalresize(data)
            return data
            
        if self.training or data.shape[0]!=1:
            print("it is impossible to train on too large tile or to do the inference on a large batch of large images")
            quit()
        with torch.no_grad():
            device = data.device
            globalresize = nn.AdaptiveAvgPool2d((data.shape[2],data.shape[3]))
            power2resize = nn.AdaptiveAvgPool2d((min(128,(data.shape[2]//32)*32),min(128,(data.shape[3]//32)*32)))
            
            data = power2resize(data)
            
            output = torch.zeros(1,datasetdescription.nbclasses,data.shape[2],data.shape[3]).cpu()
            for row in range(0,data.shape[2]-127,32):
                for col in range(0,data.shape[3]-127,32):
                    output[:,:,row:row+128,col:col+128] += self.simpleforward(data[:,:,row:row+128,col:col+128],datasetdescription.hash).cpu()
            
            return output.to(device)
        
