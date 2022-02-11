# single dataset baseline

provides code for simple binary segmentation on dataset with strict structure

## usage

python3 run.py [path]
- run train.py on [path] which produces a model model.pth on a local build folder - using only data from [path/train/]
- run test.py which uses the model on [path/test/]

## data structure

[path] should be the path to a dataset
- [path] should contain 2 folders [train] and [test]
- each [train], [test] should contain a sequence of pairs of image/mask
- image names should be {i}_x.png and mask name should be {i}_y.png
- with i a range of integer starting from 0
- *all other data are ignored* (in particular if there is "0_x.png, 1_x.png, 3_x.png" then "3_x.png" is ignored as the range stop at 1 but if 2_x.png is added then all images are used)
- mask should be image 0 corresponds to background and other value to the class of interest
- you could find in potsdam.zip a post processed version of the famous ISPRS POTSDAM dataset with suitable data structure

## feature

training relies on crops from images

testing processes the hole images by tilling WITH OVERLAP

in thread code, cropextractor is threaded so 2 different cropextractors can be used on 2 datasets having very different size
**however** this comes with worse running time compared to torchvision dataloader - and - sampling with lower "randomness"

in torchvision code, first crop are extracted, then pushed into a classical pytorch dataloader

## typical output on Potsdam

load data
define model
train
tensor([1.4680], device='cuda:0')
tensor([0.6558], device='cuda:0')
tensor([0.4999], device='cuda:0')
tensor([0.4049], device='cuda:0')
tensor([0.3506], device='cuda:0')
tensor([0.3031], device='cuda:0')
tensor([0.2750], device='cuda:0')
tensor([0.2678], device='cuda:0')
tensor([0.2412], device='cuda:0')
tensor([0.2263], device='cuda:0')
999 perf tensor([81.8246, 92.2427])
tensor([0.2115], device='cuda:0')
tensor([0.2054], device='cuda:0')
tensor([0.1975], device='cuda:0')
tensor([0.1916], device='cuda:0')
tensor([0.1727], device='cuda:0')
tensor([0.1735], device='cuda:0')
tensor([0.1644], device='cuda:0')
tensor([0.1611], device='cuda:0')
tensor([0.1548], device='cuda:0')
tensor([0.1523], device='cuda:0')
1999 perf tensor([93.5353, 97.5884])
training stops after reaching high training accuracy
load data
load model
test
perf= tensor([85.9171, 93.3978])
total duration 989.1485934257507



