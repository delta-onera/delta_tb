import numpy as np
import os
from glob import glob
from tqdm import tqdm

import torch
import torchvision #import transforms

import sys
sys.path.insert(0, "../..")

from deltatb import networks
from deltatb.losses.multiscale import MultiscaleLoss
from deltatb.metrics.optical_flow import EPE
from deltatb.dataset import RegistrationDataset_Rasterio
#from deltatb.dataset.transforms import NormalizeDynamic
from deltatb.dataset import transforms
from deltatb.dataset import co_transforms
from deltatb.dataset import flow_co_transforms

from deltatb.tools.visdom_display import VisuVisdom
from backend2 import flow_to_color_tensor, upsample_output_and_evaluate
from backend2 import warp, generate_mask, nan_to_zero, mask_preprocess
from backend2 import radar_mono_preprocess, optic_gray_preprocess, SrtmFlowGenerator
from backend2 import multi_channels_preprocess

import argparse
#preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=['opt', 'rad', 'multi'])
parser.add_argument("--data-path", type=str, default='/data/S1S2_france/') # ou /data/pgodet/S1S2_france/
parser.add_argument("--arch", type=str, default='FlowNetS',
                choices=['FlowNetS', 'PWCDCNet_siamese', 'PWCDCNet_multimodal'])
parser.add_argument("--bs", type=int, default=8) #taille du batch
parser.add_argument("--nw", type=int, default=4) #nombre de coeurs cpu à utiliser pour chargement des données
parser.add_argument("--lr", type=float, default=0.0001) # learning rate
parser.add_argument("--nb-epochs", type=int, default=300) #nombre d'epochs
parser.add_argument("--nb-iter-per-epoch", type=int, default=1200)
parser.add_argument("--savedir", type=str, default="flow_results_beta")
parser.add_argument("--expname", type=str, default="delta_flow_net_recalage_optic_rgb")
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--nocuda", action="store_true")
#parser.add_argument("--saveimages", action="store_true")
parser.add_argument("--testinterval", type=int, default=5)
parser.add_argument("--display-period", type=int, default=50)
parser.add_argument("--visuvisdom", action="store_true")
parser.add_argument("--do-not-save-visu", action="store_true")
parser.add_argument("--visdomport", type=int, default=8097)
#parser.add_argument('--scheduler-step-indices', nargs='*', default=[-1], type=int,
#                    help='List of epoch indices for learning rate decay. Must be increasing. No decay if -1')
#parser.add_argument('--scheduler-factor', default=0.1, type=float,
#                    help='multiplicative factor applied on lr each step-size')
args = parser.parse_args()


#########
#Parameters
#########
mode = args.mode # 'opt', 'rad' ou 'multi'
path_train = os.path.join(args.data_path, 'TRAIN')
path_test = os.path.join(args.data_path, 'TEST')
exp_name = args.expname
save_dir = os.path.join(args.savedir, exp_name) # path the directory where to save the model and results
num_workers = args.nw
batch_size = args.bs
adam_lr = args.lr
nbr_epochs = args.nb_epochs # training epoch
nbr_iter_per_epochs = args.nb_iter_per_epoch # number of iterations for one epoch
imsize = 448 # input image size
scheduler_step_indices = [100, 150, 200]
scheduler_factor = 0.5
div_flow = 1
weight_decay = 0.0004
in_channels = 6 # nombre de canaux en entrée (2 pour du gris, 4 pour les 2 polar de radar, 6 pour les canaux rgb)
out_channels = 2
use_cuda = (not args.nocuda) and torch.cuda.is_available() # use cuda GPU

##########

args_file = os.path.join(save_dir, 'args.txt')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(args_file, 'w') as f:
    for k, v in sorted(vars(args).items()):
        f.write('{}: {}\n'.format(k, v))

#########

if use_cuda:
    torch.backends.cudnn.benchmark = True # accelerate convolution computation on GPU
    torch.cuda.set_device(args.device)

#########
#Data
#########
print("Creating data training loader...", end="", flush=True)

#flow_co_transforms.RandomRotateSimple(10),
train_co_transforms = flow_co_transforms.Compose([
                            flow_co_transforms.RandomRotateSimple(10),
                            flow_co_transforms.RandomVerticalFlip(),
                            flow_co_transforms.RandomHorizontalFlip(),
                                ])

data_transforms = torchvision.transforms.Compose([nan_to_zero,
                                        transforms.ToTensor(torch.float)])
mask_transforms = torchvision.transforms.Compose([transforms.ToTensor(torch.float)])
#my_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def get_filelist_S1S2(path, mode='opt'):
    """
    mode : 'opt', 'rad' ou 'multi'
    """
    if mode == 'multi' or mode == 'rad':
        list_S1 = sorted(glob(os.path.join(path, '*S1*.tif')))
    if mode == 'multi' or mode == 'opt':
        list_S2 = sorted(glob(os.path.join(path, '*S2*.tif')))
    list_SRTM = sorted(glob(os.path.join(path, '*SRTM*.tif')))
    list_mask = sorted(glob(os.path.join(path, '*mask1*.tif')))
    list_batches = []
    for k in range(len(list_SRTM)):
        if mode == 'multi':
            list_batches.append(([list_S1[k], list_S2[k]], list_SRTM[k],list_mask[k]))
        elif mode == 'rad':
            list_batches.append((list_S1[k], list_SRTM[k],list_mask[k]))
        elif mode == 'opt':
            list_batches.append((list_S2[k], list_SRTM[k],list_mask[k]))
    return list_batches
filelist_train = get_filelist_S1S2(path_train, mode=mode)
filelist_test = get_filelist_S1S2(path_test, mode=mode)

if mode == 'multi':
    image_preprocess = [optic_gray_preprocess, radar_mono_preprocess]
elif mode == 'rad':
    if in_channels == 2:
        image_preprocess = radar_mono_preprocess
    elif in_channels == 4:
        image_preprocess = multi_channels_preprocess
elif mode == 'opt':
    if in_channels == 2:
        image_preprocess = optic_gray_preprocess
    elif in_channels == 6:
        image_preprocess = multi_channels_preprocess

train_dataset = RegistrationDataset_Rasterio(imsize=imsize, in_channels=in_channels,
                    filelist=filelist_train, image_preprocess=image_preprocess,
                    target_preprocess=SrtmFlowGenerator(),
                    warp_fct=warp, mask_preprocess=mask_preprocess,
                    training=True, one_image_per_file=False,
                    epoch_number_of_images=nbr_iter_per_epochs * batch_size,
                    co_transforms=None,
                    input_transforms=data_transforms,
                    target_transforms=data_transforms,
                    mask_transforms=mask_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
print("Done")

#########
#Model, loss, optimizer
#########
print("Creating the model...", end="", flush=True)
#net = networks.FlowNetS(in_channels, div_flow=div_flow)
net = networks.__dict__[args.arch](input_channels=in_channels, div_flow=div_flow)
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
    display_max_flo = 80

#########
#Training
#########
def train(epoch, nbr_batches):
    nbr_batches = len(train_loader) if nbr_batches <= 0 else min(len(train_loader), nbr_batches)
    net.train()
    error = 0
    i = 0
    t = tqdm(train_loader, total=nbr_batches, ncols=100, desc="Epoch "+str(epoch))
    for img_pair_th, target_th, mask_th in t:
        if use_cuda:
            img_pair_th = [image.cuda() for image in img_pair_th]
            target_th = target_th.cuda()
            mask_th = mask_th.cuda()
            #img_pair_th = [image.cuda(async=True) for image in img_pair_th] #et pin_memory ?
            #target_th = target_th.cuda(async=True)
        output = net(img_pair_th)
        error_ = loss_fct(output, target_th, mask_vt=mask_th)
        optimizer.zero_grad()
        error_.backward()
        optimizer.step()

        i += 1
        error += error_.data
        loss = error / i

        # display TQDM
        t.set_postfix(Loss="%.3e"%float(loss))

        #display visdom
        if i % args.display_period == 0 and args.visuvisdom:
            visu.imshow(img_pair_th[0], 'Images (train)', unnormalize=True)
            color_target_flow = flow_to_color_tensor(target_th, display_max_flo / div_flow)
            visu.imshow(color_target_flow, 'Flots VT (train)')
            visu.imshow(mask_th, 'Masks VT (train)')
            for k, out in enumerate(output):
                color_output_flow = flow_to_color_tensor(out.data, display_max_flo / div_flow)
                visu.imshow(color_output_flow, 'Flots (train)[{}]'.format(k))

        #Liberation memoire:
        del output

        if i >= nbr_batches:
                break

    t.close()
    return loss.item()

def test(epoch):
    net.eval()

    with torch.no_grad():
        test_dataset = RegistrationDataset_Rasterio(imsize=imsize, in_channels=in_channels,
                            filelist=filelist_test, image_preprocess=image_preprocess,
                            target_preprocess=SrtmFlowGenerator(),
                            warp_fct=warp, mask_preprocess=mask_preprocess,
                            training=False, co_transforms=None,
                            input_transforms=data_transforms,
                            target_transforms=data_transforms,
                            mask_transforms=mask_transforms)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=num_workers)

        t = tqdm(test_loader, ncols=100, desc="Image")
        err_moy = 0

        for img_pair_th, target_th, mask_th in t:
            if use_cuda:
                img_pair_th = [image.cuda() for image in img_pair_th]
                target_th = target_th.cuda()
                mask_th = mask_th.cuda()
                #img_pair_th = [image.cuda(async=True) for image in img_pair_th]
                #target_th = target_th.cuda(async=True) # et pin_memory ds dataloader?

            # forward backward
            output = net(img_pair_th)
            err = upsample_output_and_evaluate(err_fct, output, target_th,
                                                    mask_vt=mask_th)
            #import ipdb; ipdb.set_trace()
            err_moy += err.data.item()

            t.set_postfix(EPE="%.3e"%float(err))

            #display visdom
            if args.visuvisdom:
                visu.imshow(img_pair_th[0], 'Images (test)', unnormalize=True)
                color_target_flow = flow_to_color_tensor(target_th, display_max_flo)
                visu.imshow(color_target_flow, 'Flots VT (test)')
                visu.imshow(mask_th, 'Masks VT (test)')
                for k, out in enumerate(output):
                    color_output_flow = flow_to_color_tensor(out.data, display_max_flo)
                    visu.imshow(color_output_flow, 'Flots (test)[{}]'.format(k))

        t.close()
        return err_moy / len(test_loader)

# generate filename for saving model
print("Models and logs will be saved to: {}".format(save_dir), flush=True)
os.makedirs(save_dir, exist_ok=True)

f = open(os.path.join(save_dir, "logs.txt"), "w")
f.write("Epoch  train_loss  test_epe[px]\n")
f.flush()

for epoch in range(nbr_epochs):
    scheduler.step()
    train_loss = train(epoch, nbr_iter_per_epochs)

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

if args.visuvisdom and not args.do_not_save_visu:
    visu.save()
f.close()
