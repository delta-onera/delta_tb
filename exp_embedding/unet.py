
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loadpretrained(model,correspondance,path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    for name1,name2 in correspondance:
        fw,fb = False,False
        for name, param in pretrained_dict.items():
            if name==name1+".weight" :
                model_dict[name2+".weight"].copy_(param)
                fw=True
            if name==name1+".bias" :
                model_dict[name2+".bias"].copy_(param)
                fb=True
        if (not fw) or (not fb):
            print(name2+" not found")
            quit()
    model.load_state_dict(model_dict)

class UNET(nn.Module):
    def __init__(self,nbclasses,pretrained=""):
        super(UNET, self).__init__()

        self.nbclasses = nbclasses

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

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
        
        self.final1 = nn.Conv2d(224, 128, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(128, self.nbclasses, kernel_size=3, padding=1)

        if pretrained!="":
            correspondance=[]
            correspondance.append(("features.0","conv11"))
            correspondance.append(("features.2","conv12"))
            correspondance.append(("features.5","conv21"))
            correspondance.append(("features.7","conv22"))
            correspondance.append(("features.10","conv31"))
            correspondance.append(("features.12","conv32"))
            correspondance.append(("features.14","conv33"))
            correspondance.append(("features.17","conv41"))
            correspondance.append(("features.19","conv42"))
            correspondance.append(("features.21","conv43"))
            correspondance.append(("features.24","conv51"))
            correspondance.append(("features.26","conv52"))
            correspondance.append(("features.28","conv53"))
            loadpretrained(self, correspondance, pretrained)


    def simpleforward(self, x):
        x1 = F.leaky_relu(self.conv12(F.leaky_relu(self.conv11(x))))
        globalresize = nn.AdaptiveAvgPool2d((x1.shape[2],x1.shape[3]))

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
        
        x = F.leaky_relu(self.final2(F.leaky_relu(self.final1(x1))))
        return x


    def forward(self, data,tilesize=128):
        if 128 <= data.shape[2] <= 512 and data.shape[2]%32==0 and 128 <= data.shape[3] <= 512 and data.shape[3]%32==0:
            return self.simpleforward(data)

        if data.shape[2] <= 512 and data.shape[3] <= 512:
            globalresize = nn.AdaptiveAvgPool2d((data.shape[2],data.shape[3]))
            power2resize = nn.AdaptiveAvgPool2d((max(128,(data.shape[2]//32)*32),max(128,(data.shape[3]//32)*32)))

            data = power2resize(data)
            data = self.simpleforward(data)
            data = globalresize(data)
            return data

        if self.training or data.shape[0]!=1:
            print("it is impossible to train on too large tile or to do the inference on a large batch of large images")
            quit()
        with torch.no_grad():
            device = data.device
            globalresize = nn.AdaptiveAvgPool2d((data.shape[2],data.shape[3]))
            power2resize = nn.AdaptiveAvgPool2d((max(128,(data.shape[2]//32)*32),max(128,(data.shape[3]//32)*32)))

            data = power2resize(data)

            output = torch.zeros(1,self.nbclasses,data.shape[2],data.shape[3]).cpu()
            for row in range(0,data.shape[2]-tilesize+1,32):
                for col in range(0,data.shape[3]-tilesize+1,32):
                    output[:,:,row:row+tilesize,col:col+tilesize] += self.simpleforward(data[:,:,row:row+tilesize,col:col+tilesize]).cpu()

            return globalresize(output.to(device))

