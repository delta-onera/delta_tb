"""Image Folder Data loader"""
import torch.utils.data as data
import random
import numpy as np
import torch
import numbers

from  . import globfile

import matplotlib.pyplot as plt

def apply_function_list(x, fun, **kparams):
    """Apply a function or list over a list of object, or single object."""
    if isinstance(x, list):
        y = []
        if isinstance(fun,list):
            for x_id, x_elem in enumerate(x):
                if (fun[x_id] is not None) and (x_elem is not None):
                    y.append(fun[x_id](x_elem, **kparams))
                else:
                    y.append(None)
        else:
            for x_id, x_elem in enumerate(x):
                if x_elem is not None:
                    y.append(fun(x_elem, **kparams))
                else:
                    y.append(None)
    else:
        y = fun(x)

    return y



class SegmentationDataset(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, loaded_in_memory=False,
                filelist=None, image_loader=None, target_loader=None,
                training=True,
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                return_filenames = False
                ):
        """Init function."""

        self.loaded_in_memory = loaded_in_memory
        self.imgs = filelist
        self.training = training

        # data augmentation
        self.co_transforms = co_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

        # loaders
        self.image_loader = image_loader
        self.target_loader = target_loader

        # return filenames or not
        self.return_filenames = return_filenames


    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        if self.training:

            # the data may be loaded in memory
            if self.loaded_in_memory:
                img = globfile.segmentation_global_data["training"][index][0].copy()
                target = globfile.segmentation_global_data["training"][index][1].copy()
            else:
                input_path = self.imgs[index][0]
                target_path = self.imgs[index][1]
                img = apply_function_list(input_path, self.image_loader)
                target = apply_function_list(target_path, self.target_loader)

            # apply co transforms
            if self.co_transforms is not None:
                img,target = self.co_transforms(img, target)

            # apply transforms for inputs
            if self.input_transforms is not None:
                img = apply_function_list(img, self.input_transforms)

            # apply transform for targets
            if self.target_transforms is not None:
                target = apply_function_list(target, self.target_transforms)

            if self.return_filenames:
                return img, target, self.imgs[index][0]
            else:
                return img, target


        else: # test mode

            target = -1 # must not be none

            # the data may be loaded in memory
            if self.loaded_in_memory:
                img = globfile.segmentation_global_data["test"][index][0].copy()
                if self.imgs[index][1] is not None:
                    target = globfile.segmentation_global_data["test"][index][1].copy()
            else:
                # images
                input_path = self.imgs[index][0]
                img = apply_function_list(input_path, self.image_loader)
                # target
                if self.imgs[index][1] is not None:
                    target = apply_function_list(self.imgs[index][1], self.target_loader)

            img = apply_function_list(img, np.ascontiguousarray)

            # apply transform on inputs
            if self.input_transforms is not None:
                img = apply_function_list(img, self.input_transforms)

            # apply transform for targets
            if self.target_transforms is not None and self.imgs[index][1] is not None:
                target = apply_function_list(target, self.target_transforms)

            if self.return_filenames:
                return img, target, self.imgs[index][0]
            else:
                return img, target



    def __len__(self):
        """Length."""
        return len(self.imgs)



###### NOT TESTED
class SegmentationDataset_BigImages(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self,  imsize=256, loaded_in_memory=False,
                filelist=None, image_loader=None, target_loader=None,
                training=True,
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                one_image_per_file = True,
                epoch_number_of_images=0,
                test_stride=None
                ):
        """Init function."""

        self.loaded_in_memory = loaded_in_memory
        self.imsize = imsize
        self.imgs = filelist
        self.training = training

        # data augmentation
        self.co_transforms = co_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

        # loaders
        self.image_loader = image_loader
        self.target_loader = target_loader


        self.one_image_per_file = one_image_per_file
        self.epoch_number_of_images = epoch_number_of_images

        if test_stride is None:
            self.test_stride = self.imsize//2
        else:
            self.test_stride = test_stride

        if not self.training: # Test mode
            # in test mode
            self.coords = []
            if loaded_in_memory:
                for i in range(len(globfile.segmentation_global_data["test"])):
                    shape = globfile.segmentation_global_data["test"][i][0].shape
                    for x in range(0, shape[0]-self.imsize, self.test_stride):
                        for y in range(0, shape[1]-self.imsize, self.test_stride):
                            self.coords.append([i,x,y])
                        self.coords.append([i,x,shape[1]-self.imsize])
                    x = shape[0]-self.imsize
                    for y in range(0, shape[1]-self.imsize, self.test_stride):
                       self.coords.append([i,x,y])
                    self.coords.append([i, x,shape[1]-self.imsize])
            else:
                for im_id in range(len(self.imgs)):
                    input_path = self.imgs[im_id][0]
                    shape = self.image_loader(input_path).shape
                    for x in range(0, shape[0]-self.imsize, self.test_stride):
                        for y in range(0, shape[1]-self.imsize, self.test_stride):
                            self.coords.append([im_id,x,y])
                        self.coords.append([im_id,x,shape[1]-self.imsize])
                    x = shape[0]-self.imsize
                    for y in range(0, shape[1]-self.imsize, self.test_stride):
                       self.coords.append([im_id,x,y])
                    self.coords.append([im_id, x,shape[1]-self.imsize])

        self.current_img = None
        self.current_img_id = None

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        if self.training:

            # the data may be loaded in memory
            if self.loaded_in_memory:
                if self.one_image_per_file:
                    img = globfile.segmentation_global_data["training"][index][0].copy()
                    target = globfile.segmentation_global_data["training"][index][1].copy()
                else:
                    img_id = random.randint(0,len(self.imgs)-1)
                    img = globfile.segmentation_global_data["training"][img_id][0].copy()
                    target = globfile.segmentation_global_data["training"][img_id][1].copy()
            else:
                if self.one_image_per_file:
                    input_path = self.imgs[index][0]
                    target_path = self.imgs[index][1]
                else:
                    img_id = random.randint(0,len(self.imgs)-1)
                    input_path = self.imgs[img_id][0]
                    target_path = self.imgs[img_id][1]
                img = self.image_loader(input_path)
                target = self.target_loader(target_path)

        else: # test mode

            # get coordinates
            coord = self.coords[index]

            img_id = coord[0]
            x = coord[1]
            y = coord[2]

            # the data may be loaded in memory
            if self.loaded_in_memory:
                if self.current_img_id is None or self.current_img_id != img_id:
                    self.current_img = globfile.segmentation_global_data["test"][img_id][0].copy()
                    self.current_img_id = img_id
                img = self.current_img[x:x+self.imsize, y:y+self.imsize,:]
                shape = self.current_img.shape[:2]
            else:
                input_path = self.imgs[img_id][0]
                img = self.image_loader(input_path)
                shape = img.shape[:2]
                img = img[x:x+self.imsize, y:y+self.imsize,:]

            target = torch.LongTensor([img_id, x,y, shape[0], shape[1]])
            img = np.ascontiguousarray(img)

        if self.co_transforms is not None:
            img,target = self.co_transforms(img, target)


        if self.input_transforms is not None:
            img = self.input_transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target



    def __len__(self):
        """Length."""
        if self.training:
            if self.one_image_per_file:
                return len(self.imgs)
            else:
                return self.epoch_number_of_images
        else:
            return len(self.coords)


class RegistrationDataset_BigImages(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, big_img_size, imsize=256,
                filelist=None,
                image_loader=None, target_loader=None,
                warp_fct=None, mask_generator=None,
                training=True,
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                one_image_per_file = True,
                epoch_number_of_images=0,
                test_stride=None
                ):
        """Init function."""

        if isinstance(imsize, numbers.Number):
            self.imsize = (int(imsize), int(imsize))
        else:
            self.imsize = imsize
        if isinstance(big_img_size, numbers.Number):
            self.big_img_size = (int(big_img_size), int(big_img_size))
        else:
            self.big_img_size = big_img_size
        self.imgs = filelist
        self.training = training

        # data augmentation
        self.co_transforms = co_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

        # loaders
        self.image_loader = image_loader
        self.target_loader = target_loader

        self.mask_generator = mask_generator
        self.warp_fct = warp_fct

        self.one_image_per_file = one_image_per_file
        self.epoch_number_of_images = epoch_number_of_images

        if test_stride is None:
            self.test_stride = self.imsize//2
        else:
            self.test_stride = test_stride

        if not self.training: # Test mode
            # in test mode
            self.coords = []
            for im_id in range(len(self.imgs)):
                input_path = self.imgs[im_id][0]
                if isinstance(input_path, list):
                    shape = self.image_loader(input_path[0]).shape
                else:
                    shape = self.image_loader(input_path).shape
                for x in range(0, shape[1]-self.imsize[1], self.test_stride):
                    for y in range(0, shape[0]-self.imsize[0], self.test_stride):
                        self.coords.append([im_id,x,y])
                    self.coords.append([im_id,x,shape[0]-self.imsize[0]])
                x = shape[1]-self.imsize[1]
                for y in range(0, shape[0]-self.imsize[0], self.test_stride):
                   self.coords.append([im_id,x,y])
                self.coords.append([im_id, x,shape[0]-self.imsize[0]])

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        if self.training:
            if self.one_image_per_file:
                input_path = self.imgs[index][0]
                target_path = self.imgs[index][1]
                x = self.big_img_size[1] // 2 - self.imsize[1] // 2
                y = self.big_img_size[0] // 2 - self.imsize[0] // 2
            else:
                img_id = random.randint(0,len(self.imgs)-1)
                x = random.randint(0, self.big_img_size[1] - self.imsize[1] - 1)
                y = random.randint(0, self.big_img_size[0] - self.imsize[0] - 1)
                input_path = self.imgs[img_id][0]
                target_path = self.imgs[img_id][1]

        else: # test mode
            # get coordinates
            coord = self.coords[index]
            img_id = coord[0]
            x = coord[1]
            y = coord[2]
            input_path = self.imgs[img_id][0]
            target_path = self.imgs[img_id][1]

        img = apply_function_list(input_path, self.image_loader, x=x, y=y)
        target = apply_function_list(target_path, self.target_loader, x=x, y=y)

        if isinstance(target, list):
            flow = target[0]
        else:
            flow = target
        if isinstance(img, list):
            if not self.warp_fct is None:
                img[1] = self.warp_fct(img[1], flow)
            else:
                pass #On suppose que la deuxième image est déjà décalée donc on ne fait rien
        else:
            img = [img, self.warp_fct(img, flow)]
        if isinstance(target, list):
            pass # on suppose que target[1] est un masque préchargé
        elif not self.mask_generator is None:
            mask = self.mask_generator(img, flow)
            target = [target, mask]

        img = apply_function_list(img, np.ascontiguousarray)
        target = apply_function_list(target, np.ascontiguousarray)

        if self.co_transforms is not None:
            img,target = self.co_transforms(img, target)

        if self.input_transforms is not None:
            img = apply_function_list(img, self.input_transforms)

        if self.target_transforms is not None:
            target = apply_function_list(target, self.target_transforms)

        return img, target


    def __len__(self):
        """Length."""
        if self.training:
            if self.one_image_per_file:
                return len(self.imgs)
            else:
                return self.epoch_number_of_images
        else:
            return len(self.coords)
