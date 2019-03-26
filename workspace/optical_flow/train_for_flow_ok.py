import numpy as np
import os
from glob import glob
from tqdm import tqdm

import torch
from torchvision import transforms

import sys
sys.path.insert(0, "../..")

from deltatb import networks
from deltatb.losses.multiscale import MultiscaleLoss
from deltatb.metrics.optical_flow import EPE
from deltatb.dataset.datasets import SegmentationDataset
from deltatb.dataset.transforms import NormalizeDynamic
from deltatb.dataset import co_transforms
from deltatb.dataset import flow_co_transforms

from deltatb.tools.visdom_display import VisuVisdom
from backend import flow_to_color_tensor, upsample_output_and_evaluate
from backend import image_loader_gray, flow_loader
from backend import RandomTranslate, RandomRotate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=8) #taille du batch
parser.add_argument("--nw", type=int, default=2) #nombre de coeurs cpu à utiliser pour chargement des données
parser.add_argument("--lr", type=float, default=0.0001) # learning rate
parser.add_argument("--nb-epochs", type=int, default=300) #nombre d'epochs
parser.add_argument("--savedir", type=str, default="flow_results")
parser.add_argument("--expname", type=str, default="delta_flow_net_flying_chairs")
parser.add_argument("--nocuda", action="store_true")
#parser.add_argument("--saveimages", action="store_true")
parser.add_argument("--testinterval", type=int, default=5)
parser.add_argument("--visuvisdom", action="store_true")
parser.add_argument("--visdomport", type=int, default=8097)
#parser.add_argument('--scheduler-step-indices', nargs='*', default=[-1], type=int,
#                    help='List of epoch indices for learning rate decay. Must be increasing. No decay if -1')
#parser.add_argument('--scheduler-factor', default=0.1, type=float,
#                    help='multiplicative factor applied on lr each step-size')
args = parser.parse_args()


#########
#Parameters
#########
exp_name = args.expname
save_dir = os.path.join(args.savedir, exp_name) # path the directory where to save the model and results
num_workers = args.nw
batch_size = args.bs
adam_lr = args.lr
nbr_epochs = args.nb_epochs # training epoch
nbr_iter_per_epochs = 10000 # number of iterations for one epoch
imsize = 320, 448 # input image size
scheduler_step_indices = [100, 150, 200]
scheduler_factor = 0.5
div_flow = 20
weight_decay = 0.0004
in_channels = 2
out_channels = 2
use_cuda = (not args.nocuda) # use cuda GPU

#########

if use_cuda:
    torch.backends.cudnn.benchmark = True # accelerate convolution computation on GPU

#########
#Data
#########
print("Creating data training loader...", end="", flush=True)

train_co_transforms = co_transforms.Compose([
                            flow_co_transforms.RandomTranslate(10),
                            flow_co_transforms.RandomRotate(10,5),
                            flow_co_transforms.RandomCrop(imsize),
                            flow_co_transforms.RandomVerticalFlip(),
                            flow_co_transforms.RandomHorizontalFlip(),
                                ])

input_transforms = transforms.Compose([NormalizeDynamic(3),
                                        transforms.ToTensor()])
train_target_transforms = transforms.Compose([
                                lambda target: target / div_flow,
                                transforms.ToTensor()])
test_target_transforms = transforms.Compose([transforms.ToTensor()])

path_train = '/data/FlyingChairs_splitted/TRAIN'
path_test = '/data/FlyingChairs_splitted/TEST'
def get_filelist_flyingchairs(path):
    list_imgs1 = sorted(glob(os.path.join(path, 'IMAGES/*img1.ppm')))
    list_imgs2 = sorted(glob(os.path.join(path, 'IMAGES/*img2.ppm')))
    list_flows = sorted(glob(os.path.join(path, 'FLOW/*flow.flo')))
    list_batches = []
    for k in range(len(list_imgs1)):
        list_batches.append(([list_imgs1[k], list_imgs2[k]], list_flows[k]))
    return list_batches
filelist_train = get_filelist_flyingchairs(path_train)
filelist_test = get_filelist_flyingchairs(path_test)

train_dataset = SegmentationDataset(filelist=filelist_train,
                    image_loader=image_loader_gray,
                    target_loader=flow_loader,
                    training=True, co_transforms=train_co_transforms,
                    input_transforms=input_transforms,
                    target_transforms=train_target_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
print("Done")

#########
#Model, loss, optimizer
#########
print("Creating the model...", end="", flush=True)
net = networks.FlowNetS(in_channels, out_channels, div_flow=div_flow)
if use_cuda:
    net.cuda()
print("done")

loss_fct = MultiscaleLoss(EPE(mean=False))
err_fct = EPE(mean=True)

print("Creating optimizer...", end="", flush=True)
optimizer = torch.optim.Adam(net.parameters(), adam_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    scheduler_step_indices, scheduler_factor)
print("done")

#########
#Visu
#########
if args.visuvisdom:
    visu = VisuVisdom(exp_name, port=args.visdomport)
    display_max_flo = 40

#########
#Training
#########
def train(epoch, nbr_batches):
    nbr_batches = len(train_loader) if nbr_batches <= 0 else min(len(train_loader), nbr_batches)
    net.train()
    error = 0
    i = 0
    t = tqdm(train_loader, total=nbr_batches, ncols=100, desc="Epoch "+str(epoch))
    for img_pair_th, target_th in t:
        if use_cuda:
            img_pair_th = [image.cuda() for image in img_pair_th]
            target_th = target_th.cuda()
            #img_pair_th = [image.cuda(async=True) for image in img_pair_th] #et pin_memory ?
            #target_th = target_th.cuda(async=True)
        output = net(img_pair_th)
        error_ = loss_fct(output, target_th)
        optimizer.zero_grad()
        error_.backward()
        optimizer.step()

        i += 1
        error += error_.data
        loss = error / i

        # display TQDM
        t.set_postfix(Loss="%.3e"%float(loss))

        #display visdom
        if i == 1 and args.visuvisdom:
            visu.imshow(img_pair_th[0], 'Images (train)', unnormalize=True)
            color_target_flow = flow_to_color_tensor(target_th, display_max_flo / div_flow)
            visu.imshow(color_target_flow, 'Flots VT (train)')
            if False:
                visu.imshow(mask_vt, 'Mask VT (train)')
            for k, out in enumerate(output):
                color_output_flow = flow_to_color_tensor(out.data, display_max_flo / div_flow)
                visu.imshow(color_output_flow, 'Flots (train)[{}]'.format(k))

        #Liberation memoire:
        del output

        if i >= nbr_batches:
                break

    return loss.item()

def test(epoch):
    net.eval()

    with torch.no_grad():
        test_dataset = SegmentationDataset(filelist=filelist_test,
                            image_loader=image_loader_gray,
                            target_loader=flow_loader,
                            training=False, co_transforms=None,
                            input_transforms=input_transforms,
                            target_transforms=test_target_transforms)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=num_workers)

        t = tqdm(test_loader, ncols=100, desc="Image")
        err_moy = 0

        for img_pair_th, target_th in t:
            if use_cuda:
                img_pair_th = [image.cuda() for image in img_pair_th]
                target_th = target_th.cuda()
                #img_pair_th = [image.cuda(async=True) for image in img_pair_th]
                #target_th = target_th.cuda(async=True) # et pin_memory ds dataloader?

            # forward backward
            output = net(img_pair_th)
            err = upsample_output_and_evaluate(err_fct, output, target_th)
            err_moy += err.data

            t.set_postfix(EPE="%.3e"%float(err))

            #display visdom
            if args.visuvisdom:
                visu.imshow(img_pair_th[0], 'Images (test)', unnormalize=True)
                color_target_flow = flow_to_color_tensor(target_th, display_max_flo)
                visu.imshow(color_target_flow, 'Flots VT (test)')
                for k, out in enumerate(output):
                    color_output_flow = flow_to_color_tensor(out.data, display_max_flo)
                    visu.imshow(color_output_flow, 'Flots (test)[{}]'.format(k))

        return err_moy.item() / len(test_loader)

# generate filename for saving model
print("Models and logs will be saved to: {}".format(save_dir), flush=True)
os.makedirs(save_dir, exist_ok=True)

f = open(os.path.join(save_dir, "logs.txt"), "w")
f.write("Epoch  train_loss  test_epe[px]\n")
f.flush()

for epoch in range(nbr_epochs):
    scheduler.step()
    train_loss = train(epoch, nbr_iter_per_epochs / batch_size)

    # save the model
    torch.save(net.state_dict(), os.path.join(save_dir, "state_dict.pth"))

    # test
    if (epoch > 0 and (epoch+1)%args.testinterval==0) or (epoch == nbr_epochs-1):
        print("Testing {}".format(exp_name), flush=True)
        test_err = test(epoch)

        #display visdom
        if args.visuvisdom:
            visu.plot('Training Loss', epoch+1, train_loss)
            visu.plot('Validation Error', epoch+1, test_err)

        # write the logs
        f.write(str(epoch)+" ")
        f.write("%.4e "%train_loss)
        f.write("%.4f "%test_err)
        f.write("\n")
        f.flush()

f.close()
