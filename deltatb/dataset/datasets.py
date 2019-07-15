"""Image Folder Data loader"""
import torch.utils.data as data
import random
import numpy as np
import torch
import numbers

from  . import globfile

#import matplotlib.pyplot as plt

def apply_function_list(x, fun, *params):
    """Apply a function or list over a list of object, or single object."""
    if isinstance(x, list):
        y = []
        if isinstance(fun,list):
            for x_id, x_elem in enumerate(x):
                if (fun[x_id] is not None) and (x_elem is not None):
                    y.append(fun[x_id](x_elem, *params))
                else:
                    y.append(None)
        else:
            for x_id, x_elem in enumerate(x):
                if x_elem is not None:
                    y.append(fun(x_elem, *params))
                else:
                    y.append(None)
    else:
        y = fun(x, *params)

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
            #print('[index:{}] Avt co: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
            # apply co transforms
            if self.co_transforms is not None:
                img,target = self.co_transforms(img, target)
            #print('[index:{}] Ap co: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
            # apply transforms for inputs
            if self.input_transforms is not None:
                img = apply_function_list(img, self.input_transforms)
            #print('[index:{}] fin: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
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


class VideoFlowDataset(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, filelist=None, nframes=None,
                image_loader=None, target_loader=None,
                training=True,
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                return_filenames = False
                ):
        """Init function.
        nframes: can be
                an integer: same nb of frames for all samples
                a list of integers: random choice of seq length from these values
        """

        self.files = filelist
        self.nframes = nframes
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
            if type(self.nframes) in [list, tuple]:
                nframes = random.choice(self.nframes)
            else:
                nframes = self.nframes
            idx_first_frame = random.randint(0, len(self.files[index][0]) - nframes)
            delta_len = len(self.files[index][0]) - len(self.files[index][1])
            input_path = self.files[index][0][idx_first_frame:idx_first_frame+nframes]
            target_path = self.files[index][1][idx_first_frame:idx_first_frame+nframes-delta_len]
            img = apply_function_list(input_path, self.image_loader)
            target = apply_function_list(target_path, self.target_loader)
            #print('[index:{}] Avt co: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
            # apply co transforms
            if self.co_transforms is not None:
                img,target = self.co_transforms(img, target)
            #print('[index:{}] Ap co: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
            # apply transforms for inputs
            if self.input_transforms is not None:
                img = apply_function_list(img, self.input_transforms)
            #print('[index:{}] fin: shape={}, type={}, max={}'.format(index, img[0].shape, img[0].dtype, img[0].max()))
            # apply transform for targets
            if self.target_transforms is not None:
                target = apply_function_list(target, self.target_transforms)

            if self.return_filenames:
                return img, target, self.files[index][0]
            else:
                return img, target


        else: # test mode

            target = -1 # must not be none
            
            if type(self.nframes) in [list, tuple]:
                nframes = random.choice(self.nframes)
            else:
                nframes = self.nframes
            idx_first_frame = (len(self.files[index][0]) - nframes) // 2
            # images
            input_path = self.files[index][0][idx_first_frame:idx_first_frame+nframes]
            img = apply_function_list(input_path, self.image_loader)
            # target
            if self.files[index][1] is not None:
                delta_len = len(self.files[index][0]) - len(self.files[index][1])
                target_path = self.files[index][1][idx_first_frame:idx_first_frame+nframes-delta_len]
                target = apply_function_list(target_path, self.target_loader)

            img = apply_function_list(img, np.ascontiguousarray)

            # apply transform on inputs
            if self.input_transforms is not None:
                img = apply_function_list(img, self.input_transforms)

            # apply transform for targets
            if self.target_transforms is not None and self.files[index][1] is not None:
                target = apply_function_list(target, self.target_transforms)

            if self.return_filenames:
                return img, target, self.files[index][0]
            else:
                return img, target



    def __len__(self):
        """Length."""
        return len(self.files)


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
    """Main Class for Image Folder loader.
    Filelist est une liste des exemples d'entrainement.
    Chaque exemple d'entrainement doit contenir :
      - [[path_img_1, path_img_2], path_flo, path_mask] si on fournit un mask_loader
      - [[path_img_1, path_img_2], path_flo] si pas de mask ou si genération
                                                avec un mask_generator
    """

    def __init__(self, big_img_size, imsize=256,
                filelist=None,
                image_loader=None, target_loader=None,
                warp_fct=None, mask_generator=None,
                mask_loader=None,
                training=True,
                co_transforms=None,
                input_transforms=None,
                target_transforms=None,
                mask_transforms=None,
                one_image_per_file = True,
                epoch_number_of_images=0,
                test_stride=None
                ):
        """Init function."""

        if not (mask_loader is None or mask_generator is None):
            raise ValueError("""Mask_loader et mask_generator ne peuvent pas être
                                définis tous les deux """)

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
        self.mask_transforms = mask_transforms

        # loaders
        self.image_loader = image_loader
        self.target_loader = target_loader
        self.mask_loader = mask_loader

        self.mask_generator = mask_generator
        self.warp_fct = warp_fct

        self.one_image_per_file = one_image_per_file
        self.epoch_number_of_images = epoch_number_of_images

        if test_stride is None:
            self.test_stride = self.imsize[0]//2, self.imsize[1]//2
        else:
            self.test_stride = test_stride

        if not self.training: # Test mode
            # in test mode
            self.coords = []
            for im_id in range(len(self.imgs)):
                input_path = self.imgs[im_id][0]
                for x in range(0, self.big_img_size[1]-self.imsize[1], self.test_stride[1]):
                    for y in range(0, self.big_img_size[0]-self.imsize[0], self.test_stride[0]):
                        self.coords.append([im_id,x,y])
                    self.coords.append([im_id,x,self.big_img_size[0]-self.imsize[0]])
                x = self.big_img_size[1]-self.imsize[1]
                for y in range(0, self.big_img_size[0]-self.imsize[0], self.test_stride[0]):
                   self.coords.append([im_id,x,y])
                self.coords.append([im_id, x,self.big_img_size[0]-self.imsize[0]])

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        if self.training:
            if self.one_image_per_file:
                input_path = self.imgs[index][0]
                target_path = self.imgs[index][1]
                if not self.mask_loader is None:
                    mask_path = self.imgs[index][2]
                x = self.big_img_size[1] // 2 - self.imsize[1] // 2
                y = self.big_img_size[0] // 2 - self.imsize[0] // 2
            else:
                img_id = random.randint(0,len(self.imgs)-1)
                x = random.randint(0, self.big_img_size[1] - self.imsize[1] - 1)
                y = random.randint(0, self.big_img_size[0] - self.imsize[0] - 1)
                input_path = self.imgs[img_id][0]
                target_path = self.imgs[img_id][1]
                if not self.mask_loader is None:
                    mask_path = self.imgs[img_id][2]

        else: # test mode
            # get coordinates
            coord = self.coords[index]
            img_id = coord[0]
            x = coord[1]
            y = coord[2]
            input_path = self.imgs[img_id][0]
            target_path = self.imgs[img_id][1]
            if not self.mask_loader is None:
                mask_path = self.imgs[img_id][2]

        img = apply_function_list(input_path, self.image_loader, x, y)
        target = apply_function_list(target_path, self.target_loader, x, y)
        if not self.mask_loader is None:
            mask = apply_function_list(mask_path, self.mask_loader, x, y)

        if isinstance(img, list):
            if not self.warp_fct is None:
                img[0] = self.warp_fct(img[0], target)
            else:
                pass #On suppose que les images sont déjà décalées donc on ne fait rien
        else:
            img = [self.warp_fct(img, target), img]

        if not self.mask_generator is None:
            mask = self.mask_generator(img, target)

        img = apply_function_list(img, np.ascontiguousarray)
        target = apply_function_list(target, np.ascontiguousarray)
        mask = apply_function_list(mask, np.ascontiguousarray)

        if self.co_transforms is not None:
            if self.mask_generator is None and self.mask_loader is None:
                img,target = self.co_transforms(img, target)
            else:
                img, target, mask = self.co_transforms(img, target, mask)

        if self.input_transforms is not None:
            img = apply_function_list(img, self.input_transforms)

        if self.target_transforms is not None:
            target = apply_function_list(target, self.target_transforms)

        if self.mask_transforms is not None:
            mask = apply_function_list(mask, self.mask_transforms)

        if self.mask_generator is None and self.mask_loader is None:
            return img, target
        else:
            return img, target, mask


    def __len__(self):
        """Length."""
        if self.training:
            if self.one_image_per_file:
                return len(self.imgs)
            else:
                return self.epoch_number_of_images
        else:
            return len(self.coords)
