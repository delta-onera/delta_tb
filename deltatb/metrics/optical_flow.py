import torch
import numpy as np

class EPE:
    def __init__(self, mean=True):
        self.mean = mean

    def __call__(self, input_flow, target_flow, mask_vt=None):
        EPE_map = torch.norm(target_flow-input_flow,2,1)
        batch_size = EPE_map.size(0)
        if mask_vt is not None:
            mask_vt[mask_vt > 127] = 255
            #mask_scaled = torch.ByteTensor(mask_scaled.byte()) / 255
            mask_vt = mask_vt.byte() / 255
            EPE_map = torch.masked_select(EPE_map, mask_vt)
        if self.mean:
            if EPE_map.size(0) == 0:
                return 0
            else:
                return EPE_map.mean()
        else:
            return EPE_map.sum() / batch_size
