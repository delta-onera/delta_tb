import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiscaleLoss:
    def __init__(self, loss_type, weights=None, **loss_kwargs):
        self.weights = weights
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs

    def __call__(self, network_output, target_flow, mask_vt=None):
        def one_scale(output, target):

            batch_size = output.size(0)
            h = output.size(-2)
            w = output.size(-1)

            if output.ndimension() == 4:
                target_scaled = F.adaptive_avg_pool2d(target, (h, w))
            elif output.ndimension() == 5:
                target_scaled = F.adaptive_avg_pool3d(target,
                                                        (output.size(-3), h, w))
            else:
                raise NotImplementedError('Output and target tensors must have 4 or 5 dimensions')
            if mask_vt is None:
                loss = self.loss_type(output, target_scaled, **self.loss_kwargs)
            else: #attention, mask et conv3D ? Et documenter le type/la forme du mask à donner
                mask_scaled = F.adaptive_avg_pool2d(mask_vt, (h, w))
                mask_scaled[mask_scaled > 127] = 255
                #mask_scaled = torch.ByteTensor(mask_scaled.byte()) / 255
                mask_scaled = mask_scaled.byte() / 255
                loss = self.loss_type(output, target_scaled,
                                    mask_vt=mask_scaled, **self.loss_kwargs)
            return loss

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        if self.weights is None:
            self.weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        assert(len(self.weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, self.weights):
            loss += weight * one_scale(output, target_flow)
        return loss
