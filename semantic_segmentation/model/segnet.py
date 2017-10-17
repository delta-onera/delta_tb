"""Segnet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """Segnet network."""

    def __init__(self, input_nbr, label_nbr):
        """Init fields."""
        super(SegNet, self).__init__()

        self.input_nbr = input_nbr

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward method."""
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)
        size1 = x12.size()

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)
        size2 = x22.size()
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
        size3 = x33.size()

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
        size4 = x43.size()

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)
        size5 = x53.size()

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=size5)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=size4)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=size3)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=size2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=size1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def initialized_with_pretrained_weights(self):
        """Initialiaze."""
        corresp_name = {
                        "features.0.weight": "conv11.weight",
                        "features.0.bias": "conv11.bias",
                        "features.1.weight": "bn11.weight",
                        "features.1.bias": "bn11.bias",
                        "features.1.running_mean": "bn11.running_mean",
                        "features.1.running_var": "bn11.running_var",

                        "features.3.weight": "conv12.weight",
                        "features.3.bias": "conv12.bias",
                        "features.4.weight": "bn12.weight",
                        "features.4.bias": "bn12.bias",
                        "features.4.running_mean": "bn12.running_mean",
                        "features.4.running_var": "bn12.running_var",

                        "features.7.weight": "conv21.weight",
                        "features.7.bias": "conv21.bias",
                        "features.8.weight": "bn21.weight",
                        "features.8.bias": "bn21.bias",
                        "features.8.running_mean": "bn21.running_mean",
                        "features.8.running_var": "bn21.running_var",

                        "features.10.weight": "conv22.weight",
                        "features.10.bias": "conv22.bias",
                        "features.11.weight": "bn22.weight",
                        "features.11.bias": "bn22.bias",
                        "features.11.running_mean": "bn22.running_mean",
                        "features.11.running_var": "bn22.running_var",

                        # stage 3
                        "features.14.weight": "conv31.weight",
                        "features.14.bias": "conv31.bias",
                        "features.15.weight": "bn31.weight",
                        "features.15.bias": "bn31.bias",
                        "features.15.running_mean": "bn31.running_mean",
                        "features.15.running_var": "bn31.running_var",

                        "features.17.weight": "conv32.weight",
                        "features.17.bias": "conv32.bias",
                        "features.18.weight": "bn32.weight",
                        "features.18.bias": "bn32.bias",
                        "features.18.running_mean": "bn32.running_mean",
                        "features.18.running_var": "bn32.running_var",

                        "features.20.weight": "conv33.weight",
                        "features.20.bias": "conv33.bias",
                        "features.21.weight": "bn33.weight",
                        "features.21.bias": "bn33.bias",
                        "features.21.running_mean": "bn33.running_mean",
                        "features.21.running_var": "bn33.running_var",

                        # stage 4
                        "features.24.weight": "conv41.weight",
                        "features.24.bias": "conv41.bias",
                        "features.25.weight": "bn41.weight",
                        "features.25.bias": "bn41.bias",
                        "features.25.running_mean": "bn41.running_mean",
                        "features.25.running_var": "bn41.running_var",

                        "features.27.weight": "conv42.weight",
                        "features.27.bias": "conv42.bias",
                        "features.28.weight": "bn42.weight",
                        "features.28.bias": "bn42.bias",
                        "features.28.running_mean": "bn42.running_mean",
                        "features.28.running_var": "bn42.running_var",

                        "features.30.weight": "conv43.weight",
                        "features.30.bias": "conv43.bias",
                        "features.31.weight": "bn43.weight",
                        "features.31.bias": "bn43.bias",
                        "features.31.running_mean": "bn43.running_mean",
                        "features.31.running_var": "bn43.running_var",

                        # stage 5
                        "features.34.weight": "conv51.weight",
                        "features.34.bias": "conv51.bias",
                        "features.35.weight": "bn51.weight",
                        "features.35.bias": "bn51.bias",
                        "features.35.running_mean": "bn51.running_mean",
                        "features.35.running_var": "bn51.running_var",

                        "features.37.weight": "conv52.weight",
                        "features.37.bias": "conv52.bias",
                        "features.38.weight": "bn52.weight",
                        "features.38.bias": "bn52.bias",
                        "features.38.running_mean": "bn52.running_mean",
                        "features.38.running_var": "bn52.running_var",

                        "features.40.weight": "conv53.weight",
                        "features.40.bias": "conv53.bias",
                        "features.41.weight": "bn53.weight",
                        "features.41.bias": "bn53.bias",
                        "features.41.running_mean": "bn53.running_mean",
                        "features.41.running_var": "bn53.running_var",
                        }
        # load the state dict of pretrained model
        import torch.utils.model_zoo as model_zoo
        pretrained_sd = model_zoo.load_url("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")
        s_dict = self.state_dict()
        for name in pretrained_sd:
            if name not in corresp_name:
                continue
            if("features.0" not in name) or (self.input_nbr==3):
                s_dict[corresp_name[name]] = pretrained_sd[name]
        self.load_state_dict(s_dict)


    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
