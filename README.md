# Toolbox for DeLTA project

Toolbox for the [ONERA Delta project](https://delta-onera.github.io).

This toolbox provides code sample developped in the project for various applications. It will be enriched as the project goes on.

The main Delta Toolbox code is in ```deltatb``` folder.
Examples are in the ```workspace``` folder.


## [DeltaTB](./deltatb)

The Delta toolbox provides utility scripts for dense image machine learning with PyTorch.

* **Dataset** folder contains datasets classes in PyTorch format for loading multiple images and targets for dense prediction such as semantic segmentation.
* **networks** folder contains scripts for several networks, including UNet, SegNet...

## PWC-Net : install correlation custom layer (from github : NVIDIA/flownet2-pytorch)
The optical flow estimation using PWC-Net needs a custom correlation layer to be installed :
cd deltatb/networks/correlation_package
python ./setup.py install
Code from github : NVIDIA/flownet2-pytorch, modified to work on pytorch 1.0, cuda 10

## Examples

### Semantic Segmentation

Semantic segmentation example is in the workspace:
* [ISPRS 2D semantic segmentation](./workspace/isprs_vaihingen/)

## [License](LICENSE)
