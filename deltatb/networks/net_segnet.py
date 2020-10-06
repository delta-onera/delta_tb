
import os
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.hub as hub
__all__ = ["segnet"]

class SegNet(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, x):
        # Encoder block 1
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x1 = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        size1 = x.size()
        x, mask1 = self.pool(x1)
        
        # Encoder block 2
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x2 = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        size2 = x.size()
        x, mask2 = self.pool(x2)
        
        # Encoder block 3
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x3 = F.relu(self.conv3_3_bn(self.conv3_3(x)))
        size3 = x.size()
        x, mask3 = self.pool(x3)
        
        # Encoder block 4
        x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
        x4 = F.relu(self.conv4_3_bn(self.conv4_3(x)))
        size4 = x.size()
        x, mask4 = self.pool(x4)
        
        # Encoder block 5
        x = F.relu(self.conv5_1_bn(self.conv5_1(x)))
        x = F.relu(self.conv5_2_bn(self.conv5_2(x)))
        x = F.relu(self.conv5_3_bn(self.conv5_3(x)))
        size5 = x.size()
        x, mask5 = self.pool(x)
        
        # Decoder block 5
        x = self.unpool(x, mask5, output_size = size5)
        x = F.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = F.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = F.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4, output_size = size4)
        x = F.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = F.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = F.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3, output_size = size3)
        x = F.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = F.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = F.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2, output_size = size2)
        x = F.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = F.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1, output_size = size1)
        x = F.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))
        x = self.conv1_1_D(x)
        return x

    def load_pretrained_weights(self):

        vgg16_weights = hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")

        count_vgg = 0
        count_this = 0

        vggkeys = list(vgg16_weights.keys())
        thiskeys  = list(self.state_dict().keys())

        corresp_map = []

        while(True):
            vggkey = vggkeys[count_vgg]
            thiskey = thiskeys[count_this]

            if "classifier" in vggkey:
                break
            
            while vggkey.split(".")[-1] not in thiskey:
                count_this += 1
                thiskey = thiskeys[count_this]


            corresp_map.append([vggkey, thiskey])
            count_vgg+=1
            count_this += 1

        mapped_weights = self.state_dict()
        for k_vgg, k_segnet in corresp_map:
            if (self.in_channels != 3) and "features" in k_vgg and "conv1_1." not in k_segnet:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
            elif (self.in_channels == 3) and "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]

        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in Segnet !")
        except:
            print("Error VGG-16 weights in Segnet !")
            raise
    
    def load_from_filename(self, model_path):
        """Load weights from filename."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)


def segnet(in_channels, out_channels, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SegNet(in_channels, out_channels)
    if pretrained:
        model.load_pretrained_weights()
    return model