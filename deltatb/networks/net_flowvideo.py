import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import math
import numpy as np
try:
    from correlation_package.modules.correlation import Correlation
except:
    try:
        from correlation_package.correlation import Correlation # or from .correlation_package.correlation import Correlation
    except:
        print("Le module de correlation n'a pas pu être importé, PWC-Net ne pourra pas être utilisé.")

from .net_pwcnet import conv, predict_flow, deconv 

__all__ = ['FlowNetStack', 'FlowNetStack_2by2', 'R1FlowNetS', 'PWCDCNetStack_2by2']

def stacked_flow_to_video_flow(stacked_flow):
    #batch_size, channels, height, width = stacked_flow.size()
    #nframes = channels // 2
    #video_flow = torch.zeros(nframes, batch_size, 2, height, width)
    #if stacked_flow.is_cuda:
    #    video_flow = video_flow.cuda()
    #video_flow[:,:,0,:,:] = stacked_flow[:,::2,:,:].transpose(0,1)
    #video_flow[:,:,1,:,:] = stacked_flow[:,1::2,:,:].transpose(0,1)
    #return video_flow
    return torch.stack([stacked_flow[:,::2,:,:].transpose(0,1), stacked_flow[:,1::2,:,:].transpose(0,1)], 2)

# ------------------------------------------------------------

class FlowNetStack(nn.Module):
    'Parameter count : 38,676,504 '
    def __init__(self, input_channels=2, output_channels=2, batch_norm=False, div_flow=1):
        super(FlowNetStack,self).__init__()

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
            flow2 = stacked_flow_to_video_flow(flow2)
            flow3 = stacked_flow_to_video_flow(flow3)
            flow4 = stacked_flow_to_video_flow(flow4)
            flow5 = stacked_flow_to_video_flow(flow5)
            flow6 = stacked_flow_to_video_flow(flow6)
            return flow2,flow3,flow4,flow5,flow6
        else:
            flow2 = stacked_flow_to_video_flow(flow2)
            return self.div_flow * flow2,

class FlowNetStack_2by2(FlowNetStack):
    'Parameter count : 38,676,504 '
    def __init__(self, input_channels=1, len_seq=7, batch_norm=False, div_flow=1):
        super(FlowNetStack_2by2,self).__init__(input_channels = input_channels * len_seq, 
                                        output_channels = 2 * (len_seq - 1), 
                                        batch_norm=batch_norm, div_flow=div_flow)


class R1FlowNetS(nn.Module):
    'Parameter count : 38,676,504 '
    def __init__(self, input_channels=1, output_channels=2, batch_norm=False, div_flow=1):
        super(R1FlowNetS,self).__init__()

        self.batch_norm = batch_norm
        self.input_channels = 2 * input_channels
        self.div_flow = div_flow

        self.conv1   = conv(self.input_channels, 64, kernel_size=7, stride=2, batch_norm=self.batch_norm)
        self.conv2   = conv(64,   128, kernel_size=5, stride=2, batch_norm=self.batch_norm)
        self.conv3   = conv(128,  256, kernel_size=5, stride=2, batch_norm=self.batch_norm)
        self.conv3_1 = conv(256,  256, batch_norm=self.batch_norm)
        self.conv4   = conv(256,  512, stride=2, batch_norm=self.batch_norm)
        self.conv4_1 = conv(512,  512, batch_norm=self.batch_norm)
        self.conv5   = conv(512,  512, stride=2, batch_norm=self.batch_norm)
        self.conv5_1 = conv(512,  512, batch_norm=self.batch_norm)
        self.conv6   = conv(512,  1024, stride=2, batch_norm=self.batch_norm)
        self.conv6_1 = conv(1024, 1024, batch_norm=self.batch_norm)

        self.deconv5 = deconv(1024+1024,512)
        self.deconv4 = deconv(1024 + output_channels, 256)
        self.deconv3 = deconv(768 + output_channels, 128)
        self.deconv2 = deconv(384 + output_channels, 64)

        self.predict_flow6 = predict_flow(1024+1024, output_channels)
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

        bs, channels, height, width = list_images[0].size()
        previous_features = torch.zeros(bs, 1024, height//64, width//64)
        if list_images[0].is_cuda:
            previous_features = previous_features.cuda()

        flow2 = []
        flow3 = []
        flow4 = []
        flow5 = []
        flow6 = []

        for i in range(len(list_images) - 1):
            x = torch.cat(list_images[i:i+2], 1)

            out_conv1 = self.conv1(x)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3_1(self.conv3(out_conv2))
            out_conv4 = self.conv4_1(self.conv4(out_conv3))
            out_conv5 = self.conv5_1(self.conv5(out_conv4))
            out_conv6 = self.conv6_1(self.conv6(out_conv5))

            features = torch.cat([previous_features, out_conv6], 1)
            previous_features = out_conv6

            flow6.append(self.predict_flow6(features))
            flow6_up    = self.upsampled_flow6_to_5(flow6[-1])
            out_deconv5 = self.deconv5(features)

            concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
            flow5.append(self.predict_flow5(concat5))
            flow5_up    = self.upsampled_flow5_to_4(flow5[-1])
            out_deconv4 = self.deconv4(concat5)

            concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
            flow4.append(self.predict_flow4(concat4))
            flow4_up    = self.upsampled_flow4_to_3(flow4[-1])
            out_deconv3 = self.deconv3(concat4)

            concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
            flow3.append(self.predict_flow3(concat3))
            flow3_up    = self.upsampled_flow3_to_2(flow3[-1])
            out_deconv2 = self.deconv2(concat3)

            concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
            flow2.append(self.predict_flow2(concat2))

        if self.training:
            flow2 = torch.stack(flow2, 0)
            flow3 = torch.stack(flow3, 0)
            flow4 = torch.stack(flow4, 0)
            flow5 = torch.stack(flow5, 0)
            flow6 = torch.stack(flow6, 0)
            return flow2,flow3,flow4,flow5,flow6
        else:
            flow2 = torch.stack(flow2, 0)
            return self.div_flow * flow2,


class PWCDCNetStack_2by2(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, input_channels=1, len_seq=7, md=4, init_method='kaiming',
                    div_flow=1, deconv_relu=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNetStack_2by2,self).__init__()
        self.input_channels = input_channels
        self.len_seq = len_seq
        self.div_flow = div_flow
        self.deconv_relu = deconv_relu

        self.enc_conv1a  = conv(int(input_channels), 16, kernel_size=3, stride=2)
        self.enc_conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.enc_conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.enc_conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.enc_conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.enc_conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.enc_conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.enc_conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.enc_conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.enc_conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.enc_conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.enc_conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.enc_conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.enc_conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.enc_conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.enc_conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.enc_conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.enc_conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.dec_leakyRELU = nn.LeakyReLU(0.1)

        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd * (self.len_seq - 1)
        self.dec_conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.dec_conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.dec_conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.dec_conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.dec_conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.dec_predict_flow6 = predict_flow(od+dd[4], 2*(self.len_seq - 1))
        self.dec_deconv6 = deconv(2*(self.len_seq - 1), 2*(self.len_seq - 1), kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)
        self.dec_upfeat6 = deconv(od+int(dd[4]), 2, kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)

        od = nd * (self.len_seq - 1) + 128 + 2 * (self.len_seq - 1) + 2
        self.dec_conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.dec_conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.dec_conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.dec_conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.dec_conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.dec_predict_flow5 = predict_flow(od+dd[4], 2*(self.len_seq - 1))
        self.dec_deconv5 = deconv(2*(self.len_seq - 1), 2*(self.len_seq - 1), kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)
        self.dec_upfeat5 = deconv(od+int(dd[4]), 2, kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)

        od = nd * (self.len_seq - 1) + 96 + 2 * (self.len_seq - 1) + 2
        self.dec_conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.dec_conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.dec_conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.dec_conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.dec_conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.dec_predict_flow4 = predict_flow(od+dd[4], 2*(self.len_seq - 1))
        self.dec_deconv4 = deconv(2*(self.len_seq - 1), 2*(self.len_seq - 1), kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)
        self.dec_upfeat4 = deconv(od+int(dd[4]), 2, kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)

        od = nd * (self.len_seq - 1) + 64 + 2 * (self.len_seq - 1) + 2
        self.dec_conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.dec_conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.dec_conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.dec_conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.dec_conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.dec_predict_flow3 = predict_flow(od+dd[4], 2*(self.len_seq - 1))
        self.dec_deconv3 = deconv(2*(self.len_seq - 1), 2*(self.len_seq - 1), kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)
        self.dec_upfeat3 = deconv(od+int(dd[4]), 2, kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)

        od = nd * (self.len_seq - 1) + 32 + 2 * (self.len_seq - 1) + 2
        self.dec_conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.dec_conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.dec_conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.dec_conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.dec_conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.dec_predict_flow2 = predict_flow(od+dd[4], 2*(self.len_seq - 1))
        self.dec_deconv2 = deconv(2*(self.len_seq - 1), 2*(self.len_seq - 1), kernel_size=4, stride=2, padding=1, relu=self.deconv_relu)

        self.dec_dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dec_dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dec_dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dec_dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dec_dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dec_dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dec_dc_conv7 = predict_flow(32, 2*(self.len_seq - 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.uniform_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask


    def forward(self, list_images, ref=None):
        if ref is None:
            ref = (len(list_images) - 1) // 2

        list_c1 = []
        list_c2 = []
        list_c3 = []
        list_c4 = []
        list_c5 = []
        list_c6 = []
        
        for im in list_images:
            #c1 = self.enc_conv1b(self.enc_conv1aa(self.enc_conv1a(im)))
            list_c1.append(self.enc_conv1b(self.enc_conv1aa(self.enc_conv1a(im))))
            list_c2.append(self.enc_conv2b(self.enc_conv2aa(self.enc_conv2a(list_c1[-1]))))
            list_c3.append(self.enc_conv3b(self.enc_conv3aa(self.enc_conv3a(list_c2[-1]))))
            list_c4.append(self.enc_conv4b(self.enc_conv4aa(self.enc_conv4a(list_c3[-1]))))
            list_c5.append(self.enc_conv5b(self.enc_conv5aa(self.enc_conv5a(list_c4[-1]))))
            list_c6.append(self.enc_conv6b(self.enc_conv6a(self.enc_conv6aa(list_c5[-1]))))


        corr6 = []
        for i in range(len(list_images)-1):
            corr6.append(self.dec_leakyRELU(self.corr(list_c6[i], list_c6[i+1])))

        x = torch.cat(corr6,1)
        x = torch.cat((self.dec_conv6_0(x), x),1)
        x = torch.cat((self.dec_conv6_1(x), x),1)
        x = torch.cat((self.dec_conv6_2(x), x),1)
        x = torch.cat((self.dec_conv6_3(x), x),1)
        x = torch.cat((self.dec_conv6_4(x), x),1)
        flow6 = self.dec_predict_flow6(x)
        up_flow6 = self.dec_deconv6(flow6)
        up_feat6 = self.dec_upfeat6(x)


        corr5 = []
        for i in range(len(list_images)-1):
            corr5.append(self.dec_leakyRELU(self.corr(list_c5[i], 
                self.warp(list_c5[i+1], up_flow6[:,2*i:2*i+2,:,:] * self.div_flow / 32.0))))
        
        x = torch.cat((torch.cat(corr5,1), list_c5[ref], up_flow6, up_feat6), 1)
        x = torch.cat((self.dec_conv5_0(x), x),1)
        x = torch.cat((self.dec_conv5_1(x), x),1)
        x = torch.cat((self.dec_conv5_2(x), x),1)
        x = torch.cat((self.dec_conv5_3(x), x),1)
        x = torch.cat((self.dec_conv5_4(x), x),1)
        flow5 = self.dec_predict_flow5(x)
        up_flow5 = self.dec_deconv5(flow5)
        up_feat5 = self.dec_upfeat5(x)


        corr4 = []
        for i in range(len(list_images)-1):
            corr4.append(self.dec_leakyRELU(self.corr(list_c4[i], 
                self.warp(list_c4[i+1], up_flow5[:,2*i:2*i+2,:,:] * self.div_flow / 16.0))))
        x = torch.cat((torch.cat(corr4,1), list_c4[ref], up_flow5, up_feat5), 1)
        x = torch.cat((self.dec_conv4_0(x), x),1)
        x = torch.cat((self.dec_conv4_1(x), x),1)
        x = torch.cat((self.dec_conv4_2(x), x),1)
        x = torch.cat((self.dec_conv4_3(x), x),1)
        x = torch.cat((self.dec_conv4_4(x), x),1)
        flow4 = self.dec_predict_flow4(x)
        up_flow4 = self.dec_deconv4(flow4)
        up_feat4 = self.dec_upfeat4(x)


        corr3 = []
        for i in range(len(list_images)-1):
            corr3.append(self.dec_leakyRELU(self.corr(list_c3[i], 
                self.warp(list_c3[i+1], up_flow4[:,2*i:2*i+2,:,:] * self.div_flow / 8.0))))

        x = torch.cat((torch.cat(corr3,1), list_c3[ref], up_flow4, up_feat4), 1)
        x = torch.cat((self.dec_conv3_0(x), x),1)
        x = torch.cat((self.dec_conv3_1(x), x),1)
        x = torch.cat((self.dec_conv3_2(x), x),1)
        x = torch.cat((self.dec_conv3_3(x), x),1)
        x = torch.cat((self.dec_conv3_4(x), x),1)
        flow3 = self.dec_predict_flow3(x)
        up_flow3 = self.dec_deconv3(flow3)
        up_feat3 = self.dec_upfeat3(x)


        corr2 = []
        for i in range(len(list_images)-1):
            corr2.append(self.dec_leakyRELU(self.corr(list_c2[i], 
                self.warp(list_c2[i+1], up_flow3[:,2*i:2*i+2,:,:] * self.div_flow / 4.0))))
        x = torch.cat((torch.cat(corr2,1), list_c2[ref], up_flow3, up_feat3), 1)
        x = torch.cat((self.dec_conv2_0(x), x),1)
        x = torch.cat((self.dec_conv2_1(x), x),1)
        x = torch.cat((self.dec_conv2_2(x), x),1)
        x = torch.cat((self.dec_conv2_3(x), x),1)
        x = torch.cat((self.dec_conv2_4(x), x),1)
        flow2 = self.dec_predict_flow2(x)

        x = self.dec_dc_conv4(self.dec_dc_conv3(self.dec_dc_conv2(self.dec_dc_conv1(x))))
        flow2 += self.dec_dc_conv7(self.dec_dc_conv6(self.dec_dc_conv5(x)))

        if self.training:
            flow2 = stacked_flow_to_video_flow(flow2)
            flow3 = stacked_flow_to_video_flow(flow3)
            flow4 = stacked_flow_to_video_flow(flow4)
            flow5 = stacked_flow_to_video_flow(flow5)
            flow6 = stacked_flow_to_video_flow(flow6)
            return flow2,flow3,flow4,flow5,flow6
        else:
            flow2 = stacked_flow_to_video_flow(flow2)
            return self.div_flow * flow2,