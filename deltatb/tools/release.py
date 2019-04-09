import os
import sys

import torch
import numpy as np

import numbers

from .. import networks
from ..dataset.transforms import NormalizeDynamic

def get_n_params(model):
    pp=0
    for p in model.parameters():
        nn=1
        for s in p.size():
            nn = nn*s
        pp += nn
    return pp

class CenterCrop(object):
    """Crops the given input array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, Im):
        """
        Im : numpy.array, une image
        """
        h, w, _ = Im.shape
        th, tw = self.size

        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        croped = Im[y: y + th, x: x + tw]
        return croped

class CenterZeroPad(object):
    """Pads around the given input array to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, Im):
        """
        Im : numpy.array, une image
        """
        h, w, c = Im.shape
        th, tw = self.size
        x = int(round((tw - w) / 2.))
        y = int(round((th - h) / 2.))

        padded = np.zeros((th,tw,c), dtype=Im.dtype)
        padded[y: y + h, x: x + w] = Im
        return padded

#################################################################################

class AlgoFlowDL:
    def __init__(self, arch_name, pretrained_path, use_cuda=True,
                    upsample_flow=4, device=0, **arch_params):
        torch.cuda.set_device(device)
        #if not 'div_flow' in arch_params.keys():
        #    arch_params['div_flow'] = 20

        self.upsample_flow = upsample_flow
        self.use_cuda = use_cuda

        pretrained_data = torch.load(pretrained_path,
                                    map_location=lambda storage,
                                    loc: storage.cuda(device))
        self.net = networks.__dict__[arch_name](**arch_params)

        if 'model_state_dict' in pretrained_data.keys():
            self.net.load_state_dict(pretrained_data['model_state_dict'])
        elif 'state_dict' in pretrained_data.keys():
            self.net.load_state_dict(pretrained_data['state_dict'])
        else:
            self.net.load_state_dict(pretrained_data)

        if self.use_cuda:
            self.net.cuda()
        self.net.eval()

        self.normalize = NormalizeDynamic(3)

    def __call__(self, I1, I2):
        """
        I1 et I2 sont des np.arrays à 2 ou 3 dimensions.
        Si 3 dimensions : rows,cols,channels
        """
        h = I1.shape[0]
        w = I1.shape[1]
        padding = CenterZeroPad((h+64-h%64,w+64-w%64)) #pour avoir des multiples de 64
        crop = CenterCrop((h,w))
        if I1.ndim == 2:
            tensor1 = np.expand_dims(I1, 2).astype(np.float32)
            tensor2 = np.expand_dims(I2, 2).astype(np.float32)
        else:
            tensor1 = I1.astype(np.float32)
            tensor2 = I2.astype(np.float32)
        tensor1 = self.normalize(tensor1)
        tensor2 = self.normalize(tensor2)
        tensor1 = padding(tensor1)
        tensor2 = padding(tensor2)
        if self.use_cuda:
            tensor1 = torch.from_numpy(tensor1.transpose(2,0,1)).cuda()
            tensor2 = torch.from_numpy(tensor2.transpose(2,0,1)).cuda()
        else:
            tensor1 = torch.from_numpy(tensor1.transpose(2,0,1))
            tensor2 = torch.from_numpy(tensor2.transpose(2,0,1))
        tensor1 = torch.unsqueeze(tensor1, 0)
        tensor2 = torch.unsqueeze(tensor2, 0)

        with torch.no_grad():
            flow = self.net([tensor1, tensor2])[0]
            if self.upsample_flow > 0:
                flow = torch.nn.functional.upsample(flow,
                                            scale_factor=self.upsample_flow)
            flow = flow[0,:,:,:].cpu().numpy().transpose(1,2,0)
            flow = crop(flow)
            u = flow[:,:,0]
            v = flow[:,:,1]

        return u, v


if __name__ == '__main__':
    import torch
    from skimage.io import imread
    from deltatb.tools import release
    import pylab as pl
    pl.interactive(True)

    path_im1 = '/data/eval_flow/sintel/final/alley_2/frame_0007.png'
    path_im2 = '/data/eval_flow/sintel/final/alley_2/frame_0008.png'
    im1 = imread(path_im1, True)
    im2 = imread(path_im2, True)

    net = release.AlgoFlowDL('PWCDCNet_archarticle', 'pretrained/pwcnet_authors/bricogray.pth.tar', input_channels=2, div_flow=20)
    flo = net(im1, im2)

    pl.figure(); pl.imshow(im1, cmap='gray')
    pl.figure(); pl.imshow(im2, cmap='gray')
    pl.figure(); pl.imshow(flo[0])
    pl.figure(); pl.imshow(flo[1])
    
