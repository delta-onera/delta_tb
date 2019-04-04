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

import warnings
try:
    import rasterio
except:
    warnings.warn("rasterio cannot be imported", Warning)
    
from scipy import ndimage
from skimage.morphology import binary_closing


def upsample_output_and_evaluate(function, output, target, **kwargs):
    if type(output) in [list, tuple]:
        output = output[0]
    h = target.size(-2)
    w = target.size(-1)
    if output.ndimension() == 4:
        upsampled_output = F.upsample(output, size=(h,w),
                                        mode='bilinear', align_corners=True)
    elif output.ndimension() == 5:
        upsampled_output = F.upsample(output,
                                        size=(output.size(-3), h, w),
                                        mode='trilinear', align_corners=True)
    else:
        raise NotImplementedError('Output and target tensors must have 4 or 5 dimensions')
    return function(upsampled_output, target, **kwargs)


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

def flow_to_color_tensor(flow_batch, max_flo=None):
    flow_hsv = []
    for w_ in flow_batch:
        w = w_.clone()
        w[np.isnan(w)] = 0
        flow_hsv.append(torch.from_numpy(flow_to_color(w.cpu().numpy(), max_flo).transpose(2,0,1)))
    return torch.stack(flow_hsv, 0)


def image_loader_gray(image_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """
    im = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
    if im.max() > 1:
        im = im / 255
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

def warp(img, flow):
    """
        warp(self, I, w):-> res
        Simple fuction to wrap the image img with motion field flow
        img : np.array [HxWx1]
        flow : np.array [HxWx2]
    """
    col, row = img.shape[1], img.shape[0]
    x, y = np.meshgrid(range(col), range(row))
    out = ndimage.map_coordinates(img[:,:,0], [y+flow[:,:,1], x+flow[:,:,0]], order=1, mode='nearest')
    return out[:,:,np.newaxis]


def normalize_img(image):
    out = image - np.nanmean(image)
    image_std = np.nanstd(image)
    if image_std != 0:
        out /= image_std
    #out[np.isnan(out)] = 0
    out = np.clip(out, -3*image_std, 3*image_std)
    return out

def nan_to_zero(image):
    out = image.copy()
    out[np.isnan(out)] = 0
    return out


def radar_mono_preprocess(image):
    image = image.astype(np.float32)
    image = (image**2).sum(axis=0)
    image = normalize_img(image)
    return image[:,:,np.newaxis]

def optic_gray_preprocess(image):
    image = image.astype(np.float32)
    image = image.sum(axis=0)
    image = normalize_img(image)
    return image[:,:,np.newaxis]

class SrtmFlowGenerator:
    def __init__(self, max_displacement=3):
        self.max_displacement = max_displacement

    def __call__(self, srtm):
        srtm = srtm[0].astype(np.float32)
        mask = binary_closing(srtm != 0, selem=np.ones((7,7)))
        srtm -= srtm.min()
        srtm_max = srtm.max()
        if srtm_max != 0:
            srtm /= srtm_max
        srtm *= self.max_displacement # deplacement max de 3 pixels
        theta = 2 * np.pi * random.random()
        u = srtm * np.cos(theta)
        v = srtm * np.sin(theta)
        u[np.logical_not(mask)] = np.nan
        v[np.logical_not(mask)] = np.nan
        flow = np.stack([u,v], axis=2)
        return flow

def generate_mask(list_img, flow):
    """
    IN:
        list_img : list of 2 np.array [HxW]
        flow : np.array [HxWx2]
    OUT :
        mask_vt : np.array [HxW] de np.uint8, 0 pour False et 255 pour True.
            Eventuellement à seuiller si on applique des opérations dessus entre temps.
    """
    #normalement on avait mis le flow à NaN pour binary_closing(srtm != 0, selem=np.ones((7,7)))
    mask_vt = np.logical_not(np.logical_or(np.isnan(flow[:,:,0]),
                                            np.isnan(flow[:,:,1])))
    mask_vt = mask_vt.astype(np.uint8)[:,:,np.newaxis] * 255
    mask_vt[np.logical_or(np.isnan(list_img[0]), list_img[0] == 0)] = 0
    mask_vt[np.logical_or(np.isnan(list_img[1]), list_img[1] == 0)] = 0
    return mask_vt
#mask = binary_closing(srtm != 0, selem=np.ones((7,7)))
#u[np.logical_not(mask)] = np.nan
#v[np.logical_not(mask)] = np.nan