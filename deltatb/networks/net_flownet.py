import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import math
import numpy as np

__all__ = ["FlowNetS"]

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding='auto',
            dilation=1, batch_norm=False):
    if padding == 'auto':
        padding = (kernel_size-1)//2
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1,inplace=False)
        )

def predict_flow(in_planes,out_planes=2,bias=True):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=bias)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True, relu=True):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.1,inplace=False)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias),
        )

# ------------------------------------------------------------

class FlowNetS(nn.Module):
    'Parameter count : 38,676,504 '
    def __init__(self, input_channels=6, output_channels=2, batch_norm=False, div_flow=1):
        super(FlowNetS,self).__init__()

        self.batch_norm = batch_norm
        self.input_channels = input_channels
        self.div_flow = div_flow

        self.conv1   = conv(input_channels,   64, kernel_size=7, stride=2, batch_norm=self.batch_norm)
        self.conv2   = conv(64,   128, kernel_size=5, stride=2, batch_norm=self.batch_norm)
        self.conv3   = conv(128,  256, kernel_size=5, stride=2, batch_norm=self.batch_norm)
        self.conv3_1 = conv(256,  256, batch_norm=self.batch_norm)
        self.conv4   = conv(256,  512, stride=2, batch_norm=self.batch_norm)
        self.conv4_1 = conv(512,  512, batch_norm=self.batch_norm)
        self.conv5   = conv(512,  512, stride=2, batch_norm=self.batch_norm)
        self.conv5_1 = conv(512,  512, batch_norm=self.batch_norm)
        self.conv6   = conv(512,  1024, stride=2, batch_norm=self.batch_norm)
        self.conv6_1 = conv(1024, 1024, batch_norm=self.batch_norm)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1024 + output_channels, 256)
        self.deconv3 = deconv(768 + output_channels, 128)
        self.deconv2 = deconv(384 + output_channels, 64)

        self.predict_flow6 = predict_flow(1024, output_channels)
        self.predict_flow5 = predict_flow(1024 + output_channels, output_channels)
        self.predict_flow4 = predict_flow(768 + output_channels, output_channels)
        self.predict_flow3 = predict_flow(384 + output_channels, output_channels)
        self.predict_flow2 = predict_flow(192 + output_channels, output_channels)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(output_channels, output_channels, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(output_channels, output_channels, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(output_channels, output_channels, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(output_channels, output_channels, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.uniform_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, list_images):
        x = torch.cat(list_images, 1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.div_flow * flow2,
