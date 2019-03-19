import torch
import numpy as np

class EPE:
    def __init__(self, mean=True):
        self.mean = mean

    def __call__(self, input_flow, target_flow, mask_vt=None):
        EPE_map = torch.norm(target_flow-input_flow,2,1)
        if mask_vt is not None:
            EPE_map = torch.masked_select(EPE_map, mask_vt)
        if self.mean:
            return EPE_map.mean()
        else:
            batch_size = EPE_map.size(0)
            return EPE_map.sum() / batch_size
