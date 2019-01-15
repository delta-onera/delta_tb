## insert parent directory in path
import sys
sys.path.insert(0, "../..")


import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import scipy.misc

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


# SEMANTIC SEGMENTATION TOOLBOX
from semSeg.dataset import SegmentationDataset_BigImages
from semSeg.dataset import globfile
from semSeg.dataset import co_transforms
import semSeg.networks as networks
import semSeg.metrics.raster as metrics

from sklearn.metrics import confusion_matrix

import argparse


###################
# Utility functions
###################

def colors_to_labels(data):
    """Convert colors to labels."""
    labels = np.zeros(data.shape[:2], dtype=int)
    colors = [[0, 0, 255], [0, 255, 0], [255, 255, 0], [0, 255, 255], [255, 255, 255], [255, 0, 0]]
    for id_col, col in enumerate(colors):
        d = (data[:, :, 0] == col[0])
        d = np.logical_and(d, (data[:, :, 1] == col[1]))
        d = np.logical_and(d, (data[:, :, 2] == col[2]))
        labels[d] = id_col
    return labels

def labels_to_colors(labels):
    """Convert labels to colors."""
    data = np.zeros(labels.shape+(3,))
    colors = [[0, 0, 255], [0, 255, 0], [255, 255, 0], [0, 255, 255], [255, 255, 255], [255, 0, 0]]
    for id_col, col in enumerate(colors):
        d = (labels == id_col)
        data[d] = col
    return data

def image_loader(image_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """
    im = np.array(Image.open(image_path), dtype=np.float32)/ 255
    return im

def target_loader(image_path):
    """Load a target image.
        Args:
            image_path
        Returns:
            A numpy matrix shape (w,h, n_channels) (if multi channels), (w,h) otherwise
            int64 if discrete label
    """
    target = np.array(Image.open(image_path)).astype(np.int64)
    target = colors_to_labels(target)
    return target

def label_image_saver(image_path, label_image):
    im = labels_to_colors(label_image)
    scipy.misc.imsave(image_path, im)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--savedir", type=str, default="results")
    parser.add_argument("--datadir", type=str, default=".")
    parser.add_argument("--inmemory", action="store_true")
    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument("--saveimages", action="store_true")
    parser.add_argument("--testinterval", type=int, default=5)
    args = parser.parse_args()

    ################
    ## Filenames
    ################

    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30'] 
    data_fname = os.path.join(args.datadir,"top/top_mosaic_09cm_area{}.tif")
    target_fname = os.path.join(args.datadir,"gts_for_participants/top_mosaic_09cm_area{}.tif")

    train_filenames = []
    for im_id in train_ids:
        data_path = data_fname.format(im_id)
        target_path = target_fname.format(im_id)
        train_filenames.append([data_path, target_path])
    test_filenames = []
    for im_id in test_ids:
        data_path = data_fname.format(im_id)
        target_path = target_fname.format(im_id)
        test_filenames.append([data_path, target_path])


    ###############
    # Parameters
    ###############
    # argument parameters
    batch_size = args.bs # batch size
    save_dir = args.savedir # path the directory where to save the model and results
    load_dataset_in_memory = args.inmemory # load the whole dataset in memory or not
    use_cuda = (not args.nocuda) # use cuda GPU


    out_channels = 6  # the number of outputs of the network is the number of labels
    in_channels = 3 # input depth
    nbr_epochs = 30 # training epoch
    imsize = 128 # input image size
    nbt_iter_per_epochs = batch_size * 100 # number of iterations for one epoch

    # SGD parameters
    sgd_milestones = [10,20,25] # steps for decreasing learning rate
    sgd_lr = 0.01
    sgd_momentum = 0.9
    sgd_gamma = 0.5

    # training data transforms for augmentation
    train_co_transforms = co_transforms.Compose([co_transforms.RandomCrop(imsize),
                                            co_transforms.RandomHorizontalFlip(),
                                            co_transforms.RandomVerticalFlip(),
                                    ])

    input_transforms = transforms.Compose([transforms.ToTensor()])


    if use_cuda:
        torch.backends.cudnn.benchmark = True # acceletate convolution computation on GPU

    # if memory is large: load the dataset in memory
    if load_dataset_in_memory:
        print("loading data in memory...", end="", flush=True)
        globfile.segmentation_global_data = {}
        globfile.segmentation_global_data["training"] = []
        for im_fname in tqdm(train_filenames):
            globfile.segmentation_global_data["training"].append([image_loader(im_fname[0]), target_loader(im_fname[1])])
        print("Done")
        

    print("Creating data training loader...", end="", flush=True)
    train_dataset = SegmentationDataset_BigImages(imsize=imsize, loaded_in_memory=load_dataset_in_memory, filelist=train_filenames,
                    image_loader=image_loader, target_loader=target_loader,
                    training=True,
                    co_transforms=train_co_transforms,
                    input_transforms=input_transforms,
                    one_image_per_file = False,
                    epoch_number_of_images=nbt_iter_per_epochs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
    print("Done")


    print("Creating the model...", end="", flush=True)
    net = networks.unet(in_channels, out_channels, pretrained=True)
    if use_cuda:
        net.cuda()
    print("done")

    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.SGD(net.parameters(), sgd_lr, momentum=sgd_momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sgd_milestones, gamma=sgd_gamma)
    print("done")



    def train(epoch, nbr_iter):
        net.train()

        global_cm = np.zeros((out_channels,out_channels))
        error = 0

        t = tqdm(train_loader, ncols=140, desc="Epoch "+str(epoch))
        for batch_th, target_th in t:

            if use_cuda:
                batch_th = batch_th.cuda()
                target_th = target_th.cuda()

            # forward backward
            optimizer.zero_grad()
            output = net.forward(batch_th)
            error_ = F.cross_entropy(output, target_th)
            error_.backward()
            optimizer.step()

            # get predictions and compute loss and accuracy
            predictions_np = np.argmax(output.cpu().data.numpy(), axis=1)
            target_np = target_th.cpu().data.numpy()

            cm = confusion_matrix(target_np.ravel(), predictions_np.ravel(), labels=list(range(out_channels)))
            global_cm += cm
            error += error_.item()

            # scores
            overall_acc = metrics.stats_overall_accuracy(global_cm)
            average_acc, _ = metrics.stats_accuracy_per_class(global_cm)
            average_iou, _ = metrics.stats_iou_per_class(global_cm)
            loss = error / global_cm.sum()

            # display TQDM
            t.set_postfix(Loss="%.3e"%float(loss), OA="%.3f" %overall_acc, avAcc="%.3f"%average_acc, avIoU="%.3f"%average_iou )

        return loss, overall_acc, average_acc, average_iou

    def test(epoch):
        net.eval()
        
        with torch.no_grad():

            global_cm = np.zeros((out_channels,out_channels))

            # iterate over the test filenames
            t = tqdm(range(len(test_filenames)), ncols=120, desc="Image")
            for im_id in t:

                # print filename
                fname = test_filenames[im_id][0].split("/")[-1]
                fname, file_extension = os.path.splitext(fname)

                # load the image in memory (or not)
                if load_dataset_in_memory:
                    globfile.segmentation_global_data["test"] = []
                    globfile.segmentation_global_data["test"].append([image_loader(test_filenames[im_id][0]), target_loader(test_filenames[im_id][1])])

                # create the dataloader
                test_dataset = SegmentationDataset_BigImages(imsize=imsize, loaded_in_memory=load_dataset_in_memory, filelist=[test_filenames[im_id]],
                        image_loader=image_loader,
                        training=False,
                        input_transforms=input_transforms,
                        )
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 

                scores = None

                for batch_th, target_th in test_loader:

                    if use_cuda:
                        batch_th = batch_th.cuda()

                    # forward backward
                    output = F.softmax(net.forward(batch_th), dim=1)
                    predictions_np = output.cpu().data.numpy()

                    target_np = target_th.cpu().data.numpy()

                    for i in range(target_np.shape[0]):
                        if scores is None:
                            scores = np.zeros((out_channels, target_np[i,3], target_np[i,4]))
                        x = target_np[i,1]
                        y = target_np[i,2]
                        scores[:,x:x+imsize, y:y+imsize] += predictions_np[i]

                
                predictions = np.argmax(scores, axis=0)

                if args.saveimages:
                    fname = os.path.join(save_dir,"images", fname+".png")
                    label_image_saver(fname, predictions)

                # scores
                if test_filenames[im_id][1] is not None:
                    # there is a target image
                    # can compute the stats
                    label_image = target_loader(test_filenames[im_id][1])
                    cm = confusion_matrix(y_true=label_image.ravel(), y_pred=predictions.ravel(), labels=list(range(out_channels)))
                    global_cm += cm

                    overall_acc = metrics.stats_overall_accuracy(global_cm)
                    average_acc, _ = metrics.stats_accuracy_per_class(global_cm)
                    average_iou, _ = metrics.stats_iou_per_class(global_cm)

                    t.set_postfix(OA="%.3f" %overall_acc, avAcc="%.3f"%average_acc, avIoU="%.3f"%average_iou )

            overall_acc = metrics.stats_overall_accuracy(global_cm)
            average_acc, _ = metrics.stats_accuracy_per_class(global_cm)
            average_iou, _ = metrics.stats_iou_per_class(global_cm)
            
            return overall_acc, average_acc, average_iou

    # generate filename for saving model and images
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    if args.saveimages:
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    f = open(os.path.join(save_dir, "logs.txt"), "w")
    for epoch in range(nbr_epochs):
        scheduler.step()
        train_loss, train_acc, train_av_acc, train_av_iou = train(epoch, nbt_iter_per_epochs)

        # save the model
        torch.save(net.state_dict(), os.path.join(save_dir, "state_dict.pth"))


        # test
        if (epoch > 0 and epoch%args.testinterval==0) or (epoch == nbr_epochs-1):
            test_acc, test_av_acc, test_av_iou = test(epoch)

            # write the logs
            f.write(str(epoch)+" ")
            f.write("%.4e "%train_loss)
            f.write("%.4f "%train_acc)
            f.write("%.4f "%train_av_acc)
            f.write("%.4f "%train_av_iou)
            f.write("%.4f "%test_acc)
            f.write("%.4f "%test_av_acc)
            f.write("%.4f "%test_av_iou)
            f.write("\n")
            f.flush()
    f.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()
