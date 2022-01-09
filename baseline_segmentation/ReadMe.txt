#singledatabaseline

singledatabaseline provides code for simple binary segmentation on dataset with strict structure

##usage

python3 run.py [path]
* run train.py on [path] which produces a model model.pth on a local build folder - using only data from [path/train/]
* run test.py which uses the model on [path/test]

## data structure

[path] should be the path to a dataset
* [path] should contain 2 folders [train] and [test]
* each [train], [test] should contain a sequence of pairs of image/mask
* image names should be {i}_x.png and mask name should be {i}_y.png
* with i a range of integer starting from 0
* *all other data are ignored* (in particular if there is "0_x.png, 1_x.png, 3_x.png" then "3_x.png" is ignored as the range stop at 1 but if 2_x.png is added then all images are used)

## feature

training relies on crops from images

testing processes the hole images by tilling WITH OVERLAP

cropextractor is threaded so 2 different cropextractors can be used on 2 datasets having very different size

**however** this comes with worse running time compared to torchvision dataloader - and - sampling with lower "randomness"

