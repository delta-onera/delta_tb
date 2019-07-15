# from __future__ import division
import random
import numpy as np
import numbers
import torch
# import types

from skimage.transform import resize
#from scipy.ndimage import convolve

#from PIL import Image
#from PIL.ImageEnhance import Contrast, Brightness

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

"""
class PILRandomBrightnessAndContrastChange(object):
    '''
    brightess_std (float between -1 and 1): std of a centered gaussian distribution from which the brightness change factor is sampled.
    contrast_bounds (2-tuple of floats between -1 and 1): bounds of an uniform distribution from which the contrast change factor is sampled.
    '''

    def __init__(self, brightess_std=None, contrast_bounds=None):
        
        self.brightess_std=brightess_std
        self.contrast_bounds=contrast_bounds

    def __call__(self, input_img):
                
        if (self.brightess_std is not None) or (self.contrast_bounds is not None):
            #if input_img.max() <= 1:
            #    input_img *= 255
            pil_img = Image.fromarray(input_img.astype(np.uint8))

        if self.brightess_std is not None:
            brightness = np.random.normal(1, self.brightess_std)
            pil_img = Brightness(pil_img).enhance(brightness)
        
        if self.contrast_bounds is not None:
            contrast = np.random.uniform(1 + self.contrast_bounds[0], 1 + self.contrast_bounds[1])
            pil_img = Contrast(pil_img).enhance(contrast)
        
        if (self.brightess_std is not None) or (self.contrast_bounds is not None):
            input_img = np.array(pil_img)

        return input_img.astype(np.float32)
"""

class RandomBrightnessChange(object):
    '''
    brightess_std (float between -1 and 1): std of a centered gaussian distribution from which the brightness change factor is sampled.
    '''

    def __init__(self, brightess_std=None):
        self.brightess_std=brightess_std

    def __call__(self, input_img):
        if self.brightess_std is not None:
            brightness = np.random.normal(1, self.brightess_std)
            input_img = (1 - brightness) * np.zeros_like(input_img) \
                        + brightness * input_img

        return input_img.astype(np.float32)

class RandomContrastChange(object):
    '''
    contrast_bounds (2-tuple of floats between -1 and 1): bounds of an uniform distribution from which the contrast change factor is sampled.
    '''

    def __init__(self, min_factor, max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, input_img):
        contrast = np.random.uniform(1 + self.min_factor, 1 + self.max_factor)
        input_img = (1 - contrast) * np.ones_like(input_img) * input_img.mean() \
                        + contrast * input_img

        return input_img.astype(np.float32)

# class RandomAsymBlur(object):
#     '''This transforms makes NaN appear, do not use in train'''
#     def __init__(self, sigma1_max, sigma2_max=1, truncate=3):
#         raise Exception("This transforms makes NaN appear, do not use in train")
#         self.sigma1_max = sigma1_max
#         self.sigma2_max = sigma2_max
#         self.truncate = truncate

#     def __call__(self, image):
#         image = image.astype(np.float32)
#         theta = np.random.uniform(0, 2*np.pi)
#         sigma1 = np.random.uniform(0, self.sigma1_max)
#         sigma2 = np.random.uniform(0, self.sigma2_max)
#         rayon = self.truncate * sigma1
#         s1 = 1 / (sigma1**2)
#         s2 = 1 / (sigma2**2)
#         X, Y = np.meshgrid(np.arange(-rayon,rayon+1), np.arange(-rayon,rayon+1))
#         gauss_kernel = (1/(2*np.pi*sigma1*sigma2)) \
#                         * np.exp(-0.5 * ( (s1*np.cos(theta)**2 + s2*np.sin(theta)**2)*X**2 \
#                                         + 2*(s2-s1)*np.cos(theta)*np.sin(theta)*X*Y \
#                                         + (s1*np.sin(theta)**2 + s2*np.cos(theta)**2)*Y**2 ) )
#         if image.ndim == 3:
#             out = np.zeros_like(image)
#             for c in range(image.shape[2]):
#                 out[:,:,c] = convolve(image[:,:,c], gauss_kernel)
#         else:
#             out = convolve(image, gauss_kernel)
#         return out.astype(np.float32)




class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, image):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        img = image * (1 + random_std) + random_mean
        img = img[:,:,random_order]

        return img