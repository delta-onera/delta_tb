# ISPRS Vaihingen semantic segmentation example

## Usage

```
python train.py --datadir path_to_dataset --savedir path_save_dir --inmemory
```

For low RAM computer, you can choose not to load all images in memory:

```
python train.py --datadir path_to_dataset --savedir path_save_dir
```