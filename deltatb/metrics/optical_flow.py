import torch
import torch.nn.functional as F
import numpy as np

class EPE:
    def __init__(self, mean=True, ignore_nan=False):
        self.mean = mean
        self.ignore_nan = ignore_nan

    def __call__(self, input_flow, target_flow, mask_vt=None):
        EPE_map = torch.norm(target_flow-input_flow,2,1)
        batch_size = EPE_map.size(0)
        if mask_vt is not None:
            mask_vt[mask_vt > 127] = 255
            #mask_scaled = torch.ByteTensor(mask_scaled.byte()) / 255
            mask_vt = mask_vt.byte() / 255
            EPE_map = torch.masked_select(EPE_map, mask_vt)
        if self.ignore_nan:
            mask_nan = ~torch.isnan(EPE_map)
            EPE_map = torch.masked_select(EPE_map, mask_nan)
        if self.mean:
            if EPE_map.size(0) == 0:
                return torch.tensor(0.0)
            else:
                return EPE_map.mean()
        else:
            return EPE_map.sum() / batch_size

class EPEGradFlo:
    def __init__(self, mean=True, ignore_nan=False):
        self.mean = mean
        self.ignore_nan = ignore_nan

    def norm_gradient(self, flow_batch):
        """
        flow_batch : une seule composante du flot [batchsize, 1, H, W]
        """
        kx = torch.tensor([[0, 0, 0],
                        [-0.5, 0, 0.5],
                        [0, 0, 0]])
        ky = torch.tensor([[0, -0.5, 0],
                        [0, 0, 0],
                        [0, 0.5, 0]])
        if flow_batch.is_cuda:
            kx = kx.cuda()
            ky = ky.cuda()
        grad_x = F.conv2d(flow_batch, kx[None,None,:,:], padding=1)
        grad_y = F.conv2d(flow_batch, ky[None,None,:,:], padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2)#.unsqueeze(1)

    def __call__(self, input_flow, target_flow, mask_vt=None):
        norm_grad_u = self.norm_gradient(target_flow[:,0,:,:].unsqueeze(1))
        norm_grad_v = self.norm_gradient(target_flow[:,1,:,:].unsqueeze(1))
        EPE_map = (norm_grad_u + norm_grad_v) * torch.norm(target_flow-input_flow,2,1)
        batch_size = EPE_map.size(0)
        if mask_vt is not None:
            mask_vt[mask_vt > 127] = 255
            #mask_scaled = torch.ByteTensor(mask_scaled.byte()) / 255
            mask_vt = mask_vt.byte() / 255
            EPE_map = torch.masked_select(EPE_map, mask_vt)
        if self.ignore_nan:
            mask_nan = ~torch.isnan(EPE_map)
            EPE_map = torch.masked_select(EPE_map, mask_nan)
        if self.mean:
            if EPE_map.size(0) == 0:
                return torch.tensor(0.0)
            else:
                return EPE_map.mean()
        else:
            return EPE_map.sum() / batch_size