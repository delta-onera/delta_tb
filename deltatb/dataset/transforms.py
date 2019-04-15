# from __future__ import division
import random
import numpy as np
import numbers
import torch
# import types

from skimage.transform import resize

class Join(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self):
        pass

    def __call__(self, inputs):

        # test if is inputs is list / numpy
        if isinstance(inputs,list):

            h, w, _ = np.array(inputs[0]).shape
            c = 0
            for inp in inputs:
                c += np.array(inp).shape[2]
            res = np.array(w,h,c)

            c = 0
            for inp in inputs:
                inp_tmp = np.array(inp)
                res[:,:,c+inp_tmp.shape[2]]  = inp_tmp
                c+=inp_tmp.shape[2]

        else: # else it is numpy
            raise Exception("Join transform ERROR")

        return res


class ToTensor(object):

    def __init__(self, datatype, divider=1):
        self.dtype = datatype
        self.divider = divider

    def __call__(self, input):

        if len(input.shape) == 3:
            return torch.from_numpy(input).permute(2,0,1).type(self.dtype) / self.divider
        elif len(input.shape) == 2:
            return torch.from_numpy(input).type(self.dtype) / self.divider
        else:
            raise Exception("ToTensor: dimension problem must be 2 or 3")

class Resize(object):

    def __init__(self, imsize):
        self.imsize = imsize

    def __call__(self, inputs):

        dtype = inputs.dtype
        inputs_ = inputs.astype(np.float)

        # ORDER 0 is nearest neighbor
        inputs_ = resize(inputs_, output_shape=self.imsize, order=0, mode='reflect')

        return inputs_.astype(dtype)


class NormalizeDynamic(object):

    def __init__(self, sigma_threshold=None):
        self.threshold = sigma_threshold

    def __call__(self, inputs):
        if len(inputs.shape)>=3 :
            inputs -= inputs.reshape(-1, inputs.shape[2]).mean(axis=0)[None,None,:]
            inputs /= inputs.reshape(-1, inputs.shape[2]).std(axis=0)[None,None,:]
            if self.threshold is not None:
                inputs[inputs >  self.threshold] = self.threshold
                inputs[inputs < -self.threshold] = -self.threshold
        else:
            raise Exception("Not implemented")
        return inputs

class RandomColorTranslation(object):

    def __init__(self, value=None):
        self.value=value

    def __call__(self, inputs):
        if self.value is not None:
            r = (random.random()*2-1) * self.value
            g = (random.random()*2-1) * self.value
            b = (random.random()*2-1) * self.value
            inputs = inputs + np.array([r,g,b])[None,None,:]
        return inputs.astype(np.float32)

class RandomNormalNoise(object):

    def __init__(self, value=None):
        self.value=value

    def __call__(self, inputs):
        if self.value is not None:
            s = random.random() * self.value
            inputs =  inputs + np.random.normal(0, s, inputs.shape)
        return inputs.astype(np.float32)
