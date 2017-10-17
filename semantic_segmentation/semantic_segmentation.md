# Semantic segmentation

This repository contains helpful code to set up a semantic segmentation pipeline.

## Models

#### Proposed models
* SegNet
* Unet

#### Interface

Each model has four methods:

* `__init__(self, input_nbr, label_nbr)`: the constructor, taking as argument the input and output channel number (output channel number is the label number).

* `forward(self, x)`: the forward method, describing how the signal flows in the network.

* `initialized_with_pretrained_weights(self)`: weight initialization using VGG16 weights from torchvision.

* `load_from_filename(self, model_path)`: loading weitghs from similar model.

## Dataloaders

`ImageFolderDenseFileLists` si a dataset in the torchvision format. It has arguments:
* the filename list of images
* the source directory: input images directory
* the target directory: target images directory
* mirror: if true, random mirror for data augmentation
