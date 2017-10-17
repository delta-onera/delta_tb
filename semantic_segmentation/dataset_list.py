"""Image Folder Data loader"""

import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageMath, ImageOps
import os
import os.path
import random


def make_dataset(input_dir, target_dir, filenames):
    """Create the dataset."""
    images = []
    # deal with multiple input

    text_file = open(filenames, 'r')
    lines = text_file.readlines()

    for filename in lines:
        filename = filename.split("\n")[0]
        item = []
        item.append(os.path.join(input_dir, filename))

        if target_dir is not None:
            item.append(os.path.join(target_dir, filename))
        else:
            item.append(None)

        images.append(item)

    return images


def pil_loader(path):
    """Load PIL images."""
    return Image.open(path)


def default_loader(path):
    """Load Default loader."""
    return pil_loader(path)


class ImageFolderDenseFileLists(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, input_root,
                target_root=None,
                filenames=None, loader=default_loader,
                 training=True, mirror=True):
        """Init function."""
        #
        # get the lists of images
        imgs = make_dataset(input_root, target_root, filenames, extensions)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + input_root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.input_root = input_root
        self.target_root = target_root
        self.imgs = imgs


        self.loader = loader
        self.training = training

        self.mirror = mirror

    def __getitem__(self, index):
        """Get item."""

        input_paths = self.imgs[index][0]
        target_path = self.imgs[index][1]

        input_img = self.loader(p)

        # random flip
        if self.mirror:
            use_mirror = random.randint(0,1)
            if(use_mirror):
                input_img = ImageOps.mirror(input_img)

        # apply transformation
        input_img = self.transform(input_img)
        transform = transforms.Compose([transforms.ToTensor()])
        input_img = transform(input_img)

        if self.training:
            target_img = self.loader(target_path)
            if(use_mirror):
                target_img = ImageOps.mirror(target_img)
            target_img = transform(target_img)
        else:
            target_img = np.array([index]) # index of the image in the filelist
        target_img = np.array(target_img)

        return input_img, target_img

    def __len__(self):
        """Length."""
        return len(self.imgs)

    def get_filename(self, id):
        """Get the filename."""
        return self.imgs[id]
