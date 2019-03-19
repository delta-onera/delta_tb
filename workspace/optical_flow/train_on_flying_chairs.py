import sys
sys.path.insert(0, "../..")

import numpy as np
from tqdm import tqdm
from PIL import Image

import os
from glob import glob

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from deltatb.dataset import co_transforms
from deltatb.dataset.transforms import NormalizeDynamic
from deltatb.dataset.datasets import SegmentationDataset
from deltatb.losses.multiscale import MultiscaleLoss
from deltatb.metrics.optical_flow import EPE
import deltatb.networks as networks

import argparse

from deltatb.tools.visdom_display import VisuVisdom
import cv2

# ----------------------------------------------------------------------

def image_loader_gray(image_path):
    """Load an image.
        Args:
            image_path
        Returns:
            A numpy float32 array shape (w,h, n_channel)
    """
    im = np.array(Image.open(image_path).convert('L'), dtype=np.float32)/ 255
    im = np.expand_dims(im, 2)
    return im

class FlowLoader:
    def __init__(self, div_flow=1):
        self.div_flow = div_flow

    def __call__(self, path):
        if path[-4:] == '.flo':
            with open(path, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                data = np.fromfile(f, np.float32, count=2*w*h)
                data /= self.div_flow
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D
        elif path[-4:] == '.pfm':
            data = readPFM(path)[0][:,:,:2]
            return data / self.div_flow
        elif path[-4:] == '.png': #kitti 2015
            import cv2
            flo_file = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
            invalid = (flo_file[:,:,0] == 0)
            flo_img = flo_img - 32768
            flo_img = flo_img / 64
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] = 0
            return(flo_img / self.div_flow)

def upsample_output_and_evaluate(function, output, target, **kwargs):
    if type(output) in [list, tuple]:
        output = output[0]
    h = target.size(-2)
    w = target.size(-1)
    if output.ndimension() == 4:
        upsampled_output = F.upsample(output, size=(h,w),
                                        mode='bilinear', align_corners=True)
    elif output.ndimension() == 5:
        upsampled_output = F.upsample(output,
                                        size=(output.size(-3), h, w),
                                        mode='trilinear', align_corners=True)
    else:
        raise NotImplementedError('Output and target tensors must have 4 or 5 dimensions')
    return function(upsampled_output, target, **kwargs)

def flow_to_color(w, maxflow=None, dark=False):
        u = w[0]
        v = w[1]
        def flow_map(N, D, mask, maxflow, n):
            cols, rows = N.shape[1], N.shape[0]
            I = np.ones([rows,cols,3])
            I[:,:,0] = np.mod((-D + np.pi/2) / (2*np.pi),1)*360
            I[:,:,1] = np.clip((N * n / maxflow),0,1)
            I[:,:,2] = np.clip(n - I[:,:,1], 0 , 1)
            return cv2.cvtColor(I.astype(np.float32),cv2.COLOR_HSV2RGB)
        cols, rows = u.shape[1], u.shape[0]
        N = np.sqrt(u**2+v**2)
        if maxflow is None:
            maxflow = np.max(N[:])
        D = np.arctan2(u,v)
        if dark:
            ret = 1 - flow_map(N,D,np.ones([rows,cols]), maxflow, 8)
        else:
            ret = flow_map(N,D,np.ones([rows,cols]), maxflow, 8)
        return ret

def flow_to_color_tensor(flow_batch, max_flo=None):
    flow_hsv = []
    for w in flow_batch:
        flow_hsv.append(torch.from_numpy(flow_to_color(w.cpu().numpy(), max_flo).transpose(2,0,1)))
    return torch.stack(flow_hsv, 0)

# --------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=8) #taille du batch
parser.add_argument("--nw", type=int, default=2) #nombre de coeurs cpu à utiliser pour chargement des données
parser.add_argument("--lr", type=float, default=0.0001) # learning rate
parser.add_argument("--savedir", type=str, default="results_on_flying_chairs")
parser.add_argument("--expname", type=str, default="flow_net")
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

###############
# Parameters
###############
# argument parameters
batch_size = args.bs # batch size
exp_name = args.expname
save_dir = os.path.join(args.savedir, exp_name) # path the directory where to save the model and results
use_cuda = (not args.nocuda) # use cuda GPU
num_workers = args.nw
adam_lr = args.lr
#scheduler_step_indices = args.scheduler_step_indices
#scheduler_factor = args.scheduler_factor

out_channels = 2  # the number of outputs of the network is 2 for a biframe flow
in_channels = 2 # 2 imgs in gray levels : 2 channels
nbr_epochs = 300 # training epoch
scheduler_step_indices = [100, 150, 200]
scheduler_factor = 0.5
nbr_iter_per_epochs = 10000 # number of iterations for one epoch
imsize = 320, 448 # input image size
weight_decay = 0.0004

div_flow = 20 #seulement appliqué pendant training.
loss_fct = MultiscaleLoss(EPE(mean=False))
err_fct = EPE(mean=True)

#display
if args.visuvisdom:
    visu = VisuVisdom(exp_name, port=args.visdomport)
    display_max_flo = 40

###############
# Données (FlyingChairs)
###############
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

###############
# Augmentation de données :
###############
train_co_transforms = co_transforms.Compose([
                                        co_transforms.RandomCrop(imsize),
                                        co_transforms.RandomHorizontalFlip(),
                                        co_transforms.RandomVerticalFlip(),
                                ])

input_transforms = transforms.Compose([NormalizeDynamic(3),
                                        transforms.ToTensor()])
target_transforms = transforms.Compose([transforms.ToTensor()])

if use_cuda:
    torch.backends.cudnn.benchmark = True # accelerate convolution computation on GPU

print("Creating data training loader...", end="", flush=True)
train_dataset = SegmentationDataset(filelist=filelist_train,
                    image_loader=image_loader_gray,
                    target_loader=FlowLoader(div_flow=div_flow),
                    training=True, co_transforms=train_co_transforms,
                    input_transforms=input_transforms, target_transforms=target_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
print("Done")


print("Creating the model...", end="", flush=True)
net = networks.FlowNetS(in_channels, out_channels, div_flow=div_flow)
if use_cuda:
    net.cuda()
print("done")

print("Creating optimizer...", end="", flush=True)
optimizer = torch.optim.Adam(net.parameters(), adam_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    scheduler_step_indices, scheduler_factor)
print("done")

def train(epoch, nbr_batches):
    net.train()

    error = 0
    i = 0

    t = tqdm(train_loader, total=nbr_batches, ncols=100, desc="Epoch "+str(epoch))
    for img_pair_th, target_th in t:

        if use_cuda:
            img_pair_th = [image.cuda() for image in img_pair_th]
            target_th = target_th.cuda()

        # forward backward
        optimizer.zero_grad()
        output = net.forward(img_pair_th)
        error_ = loss_fct(output, target_th)
        error_.backward()
        optimizer.step()

        i += 1
        error += error_.item()
        loss = error / i

        # display TQDM
        t.set_postfix(Loss="%.3e"%float(loss))

        if i == 1 and args.visuvisdom:
            #display visdom
            visu.imshow(img_pair_th[0], 'Images (train)', unnormalize=True)
            color_target_flow = flow_to_color_tensor(target_th, display_max_flo / div_flow)
            visu.imshow(color_target_flow, 'Flots VT (train)')
            if False:
                visu.imshow(mask_vt, 'Mask VT (train)')
            for k, out in enumerate(output):
                color_output_flow = flow_to_color_tensor(out.data, display_max_flo / div_flow)
                visu.imshow(color_output_flow, 'Flots (train)[{}]'.format(k))

        if i >= nbr_batches:
            break

    return loss

def test(epoch):
    net.eval()

    with torch.no_grad():

        # create the dataloader
        test_dataset = SegmentationDataset(filelist=filelist_test,
                            image_loader=image_loader_gray,
                            target_loader=FlowLoader(div_flow=1),
                            training=False, co_transforms=None,
                            input_transforms=input_transforms, target_transforms=target_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                            shuffle=False, num_workers=num_workers)
        # iterate over the test filenames
        t = tqdm(test_loader, ncols=100, desc="Image")
        err_moy = 0

        for img_pair_th, target_th in t:
            if use_cuda:
                img_pair_th = [image.cuda() for image in img_pair_th]
                target_th = target_th.cuda()

            # forward backward
            output = net.forward(img_pair_th)
            err = upsample_output_and_evaluate(err_fct, output, target_th)
            err_moy += err

            t.set_postfix(EPE="%.3e"%float(err))

            #display visdom
            if args.visuvisdom:
                visu.imshow(img_pair_th[0], 'Images (test)', unnormalize=True)
                color_target_flow = flow_to_color_tensor(target_th, display_max_flo)
                visu.imshow(color_target_flow, 'Flots VT (test)')
                for k, out in enumerate(output):
                    color_output_flow = flow_to_color_tensor(out.data, display_max_flo)
                    visu.imshow(color_output_flow, 'Flots (test)[{}]'.format(k))

        return err_moy / len(test_loader)

# generate filename for saving model and images
os.makedirs(os.path.join(save_dir), exist_ok=True)
#if args.saveimages:
#    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

f = open(os.path.join(save_dir, "logs.txt"), "w")
f.write("Epoch  train_loss  test_epe[px]\n")
f.flush()
for epoch in range(nbr_epochs):
    scheduler.step()
    train_loss = train(epoch, nbr_iter_per_epochs / batch_size)

    # save the model
    torch.save(net.state_dict(), os.path.join(save_dir, "state_dict.pth"))

    # test
    if (epoch > 0 and epoch%args.testinterval==0) or (epoch == nbr_epochs-1):
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


#if __name__ == "__main__":
#    # execute only if run as a script
#    main()
