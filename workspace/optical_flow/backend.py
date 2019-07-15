import numpy as np
import torch
import cv2
import numbers
import random

from PIL import Image

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
from scipy import ndimage

import numpy as np
import re
import sys

from deltatb import networks


def readPFM(file):
    '''
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(file, 'rb') as file:
        header = file.readline().rstrip()
        if header.decode() == 'PF':
            color = True
        elif header.decode() == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -1 * scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data, scale


def upsample_output_and_evaluate(function, output, target, **kwargs):
    if type(output) in [list, tuple]:
        output = output[0]
    h = target.size(-2)
    w = target.size(-1)
    if output.ndimension() == 4:
        upsampled_output = F.upsample(output, size=(h,w),
                                        mode='bilinear', align_corners=True)
    #elif output.ndimension() == 5:
    #    upsampled_output = F.upsample(output,
    #                                    size=(output.size(-3), h, w),
    #                                    mode='trilinear', align_corners=True)
    else:
        raise NotImplementedError('Output and target tensors must have 4 dimensions')
    return function(upsampled_output, target, **kwargs)

def upsample_output_and_evaluate_video(function, output, target, **kwargs):
    if type(output) in [list, tuple]:
        output = output[0]
    nframes = output.size(0)
    out = 0
    for n in range(nframes):
        out += upsample_output_and_evaluate(function, output[n], target[n], **kwargs)
    return out / nframes 


def flow_to_color(w, maxflow=None, dark=False):
        u = w[0]
        v = w[1]
        def flow_map(N, D, mask, maxflow, n):
            cols, rows = N.shape[1], N.shape[0]
            I = np.ones([rows,cols,3])
            I[:,:,0] = np.mod((-D + np.pi/2) / (2*np.pi),1)*360
            I[:,:,1] = np.clip((N * n / maxflow),0,1)
            I[:,:,2] = np.clip(n - I[:,:,1], 0 , 1)
            return cv2.cvtColor(I.astype(np.float32),cv2.COLOR_HSV2RGB)
        cols, rows = u.shape[1], u.shape[0]
        N = np.sqrt(u**2+v**2)
        if maxflow is None:
            maxflow = np.max(N[:])
        D = np.arctan2(u,v)
        if dark:
            ret = 1 - flow_map(N,D,np.ones([rows,cols]), maxflow, 8)
        else:
            ret = flow_map(N,D,np.ones([rows,cols]), maxflow, 8)
        return ret

# def flow_to_color_tensor(flow_batch, max_flo=None):
#     flow_hsv = []
#     for w_ in flow_batch:
#         w = w_.clone()
#         w[np.isnan(w)] = 0
#         flow_hsv.append(torch.from_numpy(flow_to_color(w.cpu().numpy(), max_flo).transpose(2,0,1)))
#     return torch.stack(flow_hsv, 0)

def flow_to_color_tensor(flow_batch, max_flo=None):
    flow_hsv = []
    for w_ in flow_batch:
        w = w_.clone().cpu().numpy()
        w[np.isnan(w)] = 0
        flow_hsv.append(torch.from_numpy(flow_to_color(w, max_flo).transpose(2,0,1)))
    return torch.stack(flow_hsv, 0)


def image_loader_gray(image_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """
    im = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
    #if im.max() > 1:
    #    im = im / 255
    im = np.expand_dims(im, 2)
    return im

def flow_loader(path):
    if path[-4:] == '.flo':
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w, 2))
        return data2D
    elif path[-4:] == '.pfm':
        data = readPFM(path)[0][:,:,:2]
        return data
    elif path[-4:] == '.png': #kitti 2015
        import cv2
        flo_file = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
        invalid = (flo_file[:,:,0] == 0)
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img

# ---------------------------------------------------

class CenterZeroPadMultiple(object):
    """Pads around the given input array to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, multiple):
        self.multiple = multiple #64

    def __call__(self, im):
        """
        im : numpy.array, une image
        """
        h, w, c = im.shape
        th, tw = (h+self.multiple-h%self.multiple,w+self.multiple-w%self.multiple)
        x = int(round((tw - w) / 2.))
        y = int(round((th - h) / 2.))

        padded_im = np.zeros((th,tw,c), dtype=im.dtype)
        padded_im[y: y + h, x: x + w] = im

        return padded_im

def load_pretrained_model(arch_name, pretrained_path, device=0, **arch_args):
    pretrained_data = torch.load(pretrained_path,
                                map_location=lambda storage,
                                loc: storage.cuda(device))

    net = networks.__dict__[arch_name](**arch_args)

    if 'model_state_dict' in pretrained_data.keys():
        net.load_state_dict(pretrained_data['model_state_dict'])
    elif 'state_dict' in pretrained_data.keys():
        net.load_state_dict(pretrained_data['state_dict'])
    else:
        net.load_state_dict(pretrained_data)

    return net

class MultiFactorMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float or list, len == len(milestone), of floats): Multiplicative factors of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        self.milestones = milestones
        if type(gammas) in [list, tuple]:
            if len(gammas) == 1:
                self.gammas = gammas * len(milestones)
            else:
                assert len(milestones) == len(gammas)
                self.gammas = gammas
        else:
            self.gammas = [gammas] * len(milestones)
        super(MultiFactorMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gammas[self.milestones.index(self.last_epoch)]
                for group in self.optimizer.param_groups]