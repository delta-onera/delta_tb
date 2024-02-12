import numpy
import PIL
from PIL import Image
import torch
import torchvision


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        del tmp[7]
        self.features = tmp.cuda()

    def forward(self, x):
        x = x.cuda() / 255.0
        x = (x - 0.5) / 0.25
        x = self.features(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x


def extractKeyCells(x, nb=300, seuil=None):
    assert len(x.shape) == 3
    assert nb is None or seuil is None
    assert nb is not None or seuil is not None

    z = x / torch.sqrt((x * x + 0.1).sum(0).unsqueeze(0))
    z = z.cuda().flatten(1)

    GRAM = torch.matmul(z.transpose(0, 1), z)
    GRAM = torch.nn.functional.softmax(GRAM, 1)
    GRAM = torch.diagonal(GRAM).cpu()

    if seuil is None:
        seuil = list(GRAM.numpy())
        seuil = float(sorted(seuil)[-nb])

    GRAM = GRAM.view(x.shape[1], x.shape[2])
    return GRAM >= seuil


with torch.no_grad():
    net = FeatureExtractor()

    image = PIL.Image.open("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")
    image = numpy.asarray(image.convert("RGB").copy())
    image = torch.Tensor(image)[0:9600, 0:9600, :].clone()
    image = torch.stack([image[:, :, 0], image[:, :, 1], image[:, :, 2]], dim=0)

    print("extract feature on native image")
    X = image.unsqueeze(0)
    Z = torch.zeros(256, 150, 150)
    for r in range(10):
        for c in range(10):
            x = X[:, :, 960 * r : 960 * (r + 1), 960 * c : 960 * (c + 1)]
            z = net(x)[0]
            Z[:, 15 * r : 15 * (r + 1), 15 * c : 15 * (c + 1)] = z

    print("extract key point on native image")
    keypoint_natif = extractKeyCells(Z)

    print("extract feature on rotated image")
    X = torch.rot90(image, dims=(1, 2)).unsqueeze(0)
    Z = torch.zeros(256, 150, 150)
    for r in range(10):
        for c in range(10):
            x = X[:, :, 960 * r : 960 * (r + 1), 960 * c : 960 * (c + 1)]
            z = net(x)[0].cpu()
            Z[:, 15 * r : 15 * (r + 1), 15 * c : 15 * (c + 1)] = z

    print("extract key point on rotated image")
    keypoint = extractKeyCells(Z)
    keypoint = torch.rot90(keypoint, k=-1)

    print("compare key point sets")
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(image, size=1200, mode="bilinear")[0]
    tmp = keypoint.view(1, 1, 150, 150).float()
    keypoint = torch.nn.functional.interpolate(tmp, size=1200, mode="bilinear")
    tmp = keypoint_natif.view(1, 1, 150, 150).float()
    keypoint_natif = torch.nn.functional.interpolate(tmp, size=1200, mode="bilinear")

    image[1] = image[1] + ((keypoint > 0) * (keypoint_natif > 0)).float() * 50
    image[0] = image[0] + ((keypoint > 0) != (keypoint_natif > 0)).float() * 50
    image = torch.clamp(image, 0, 255)
    torchvision.utils.save_image(image / 255, "build/amer.png")
