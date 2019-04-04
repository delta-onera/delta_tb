import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DilatedNet(nn.Module):
    """Learning Deep CNN Denoiser Prior for Image Restoration. Zhang et al. ICCV 2017"""

    def __init__(self, input_nbr, label_nbr, residual=True, dilations=[1,2,3,4,3,2,1]):
        """Init fields."""
        super(DilatedNet, self).__init__()

        # parameters
        self.res = residual

        # convolutions
        self.conv1 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=dilations[0], dilation=dilations[0])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=dilations[3], dilation=dilations[3])
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=dilations[4], dilation=dilations[4])
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=dilations[5], dilation=dilations[5])
        self.conv7 = nn.Conv2d(64, label_nbr, kernel_size=3, padding=dilations[6], dilation=dilations[6])
        
        # batchnorm
        batchNorm_momentum = 0.1
        self.bn2 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.bn3 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.bn4 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.bn5 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.bn6 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        # init the weights
        self.init_weights()


    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, img):
        """Forward method."""

        if self.res: # create the residual branch
            residual = img

        # Stage 1
        x = F.relu(self.conv1(img))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)

        if self.res: # add the residual branch
            return x + residual
        else:
            return x


def dilatedCNN(in_channels, out_channels, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedNet(in_channels, out_channels, kwargs)
    if pretrained:
        model.load_pretrained_weights()
    return model