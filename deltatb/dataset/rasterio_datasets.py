"""Raster Data loader"""
import torch.utils.data as data
import random
import numpy as np
import torch
import numbers

import rasterio

from .datasets import apply_function_list

class RegistrationDataset_Rasterio(data.Dataset):
    """Main Class for Image Folder loader.
    Filelist est une liste des exemples d'entrainement.
    Chaque exemple d'entrainement doit contenir :
      [[path_img_1, path_img_2], path_flo, path_mask]
    """

    def __init__(self, imsize=256,in_channels=2,
                filelist=None,
                image_preprocess=None, target_preprocess=None,
                mask_preprocess=None, mask_generator=None,
                training=True, warp_fct=None, 
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                mask_transforms=None,
                one_image_per_file = False,
                epoch_number_of_images=0,
                ):
        """Init function."""

        if not (mask_preprocess is None or mask_generator is None):
            raise ValueError("""mask_preprocess et mask_generator ne peuvent pas être
                                définis tous les deux """)

        if isinstance(imsize, numbers.Number):
            self.imsize = (int(imsize), int(imsize))
        else:
            self.imsize = imsize
        self.imgs = filelist
        self.training = training

        # data augmentation
        self.co_transforms = co_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.mask_transforms = mask_transforms

        # loaders
        self.image_preprocess = image_preprocess
        self.target_preprocess = target_preprocess
        self.mask_preprocess = mask_preprocess
        self.mask_generator = mask_generator

        self.warp_fct = warp_fct
        self.in_channels = in_channels

        self.one_image_per_file = one_image_per_file
        self.epoch_number_of_images = epoch_number_of_images

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        if self.training and not self.one_image_per_file:
            img_id = random.randint(0,len(self.imgs)-1)
            input_path = self.imgs[img_id][0]
            target_path = self.imgs[img_id][1]
            input_src = apply_function_list(input_path, rasterio.open)
            target_src = rasterio.open(target_path)
            if not self.mask_preprocess is None:
                mask_path = self.imgs[img_id][2]
                mask_src = rasterio.open(mask_path)
            
            if isinstance(input_src, list):
                big_img_size = input_src[0].height, input_src[0].width
            else:
                big_img_size = input_src.height, input_src.width

            x = random.randint(0, big_img_size[1] - self.imsize[1] - 1)
            y = random.randint(0, big_img_size[0] - self.imsize[0] - 1)

        else: # test mode or self.one_image_per_file
            input_path = self.imgs[index][0]
            target_path = self.imgs[index][1]
            input_src = apply_function_list(input_path, rasterio.open)
            target_src = rasterio.open(target_path)
            if not self.mask_preprocess is None:
                mask_path = self.imgs[index][2]
                mask_src = rasterio.open(mask_path)

            if isinstance(input_src, list):
                big_img_size = input_src[0].height, input_src[0].width
            else:
                big_img_size = input_src.height, input_src.width

            x = big_img_size[1] // 2 - self.imsize[1] // 2
            y = big_img_size[0] // 2 - self.imsize[0] // 2

        window = rasterio.windows.Window(x, y, self.imsize[1], self.imsize[0])
        if isinstance(input_src, list):
            img = [src.read(window=window) for src in input_src]
            for src in input_src:
                src.close()
        else:
            img = input_src.read(window=window)
            # print(np.shape(img))
            # print(img.dtype)
            # print(img.min(),img.max())
            input_src.close()
        img = apply_function_list(img, self.image_preprocess)
        target = self.target_preprocess(target_src.read(window=window))
        target_src.close()
        if not self.mask_preprocess is None:
            mask = self.mask_preprocess(mask_src.read(window=window))
            # print(np.shape(mask))
            # print(mask.min(),mask.max())
            # print(mask.dtype)
            mask_src.close()

        if isinstance(img, list):
            if not self.warp_fct is None:
                img[0] = self.warp_fct(img[0], target, self.in_channels)
            else:
                pass #On suppose que les images sont déjà décalées donc on ne fait rien
        else:
            img = [self.warp_fct(img, target, self.in_channels), img]

        if not self.mask_generator is None:
            mask = self.mask_generator(img, target)

        img = apply_function_list(img, np.ascontiguousarray)
        target = apply_function_list(target, np.ascontiguousarray)
        if (not self.mask_preprocess is None) or (not self.mask_generator is None):
            mask = apply_function_list(mask, np.ascontiguousarray)

        if self.co_transforms is not None:
            if (not self.mask_preprocess is None) or (not self.mask_generator is None):
                img, target, mask = self.co_transforms(img, target, mask)
            else:
                img,target = self.co_transforms(img, target)

        if self.input_transforms is not None:
            img = apply_function_list(img, self.input_transforms)

        if self.target_transforms is not None:
            target = apply_function_list(target, self.target_transforms)

        if self.mask_transforms is not None:
            mask = apply_function_list(mask, self.mask_transforms)

        if (not self.mask_preprocess is None) or (not self.mask_generator is None):
            return img, target, mask
        else:
            return img, target


    def __len__(self):
        """Length."""
        if self.training:
            if self.one_image_per_file:
                return len(self.imgs)
            else:
                return self.epoch_number_of_images
        else:
            return len(self.imgs)
