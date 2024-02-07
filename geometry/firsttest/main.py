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
        del tmp[6]
        del tmp[5]
        del tmp[4]
        self.features = tmp.cuda().half()

    def forward(self, x):
        with torch.no_grad():
            x = x.cuda().half() / 255.0
            x = (x - 0.5) / 0.25
            return self.features(x)


net = FeatureExtractor()

image = PIL.Image.open("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")
image = numpy.asarray(image.convert("RGB").copy())
image = torch.Tensor(image).clone()
image = torch.stack([image[:, :, 0], image[:, :, 1], image[:, :, 2]], dim=0)
image = image.unsqueeze(0)

allfeatures = torch.zeros(64, 1250, 1250)
for r in range(10):
    for c in range(10):
        x = image[:, :, 1000 * r : 1000 * (r + 1), 1000 * c : 1000 * (c + 1)]
        z = net(x)[0].cpu()
        allfeatures[:, 125 * r : 125 * (r + 1), 125 * c : 125 * (c + 1)] = z

print("extract stats")
allfeatures.cuda()
mu = allfeatures.flatten(1).mean(1)
allfeatures -= mu.view(64, 1, 1)
sig = torch.sqrt((allfeatures.flatten(1) ** 2).mean(1))
allfeatures /= sig.view(64, 1, 1)

normalizednorms = allfeatures.abs().mean(0)

seuil = list(normalizednorms.flatten().cpu().numpy())
seuil = sorted(seuil)
seuil = seuil[90 * len(seuil) // 100]

image1250 = torch.nn.functional.interpolate(image, size=1250, mode="bilinear")
image1250 = image1250[0] / 255
torchvision.utils.save_image(image1250, "build/image.png")

image1250 *= (normalizednorms > seuil).float().unsqueeze(0)
torchvision.utils.save_image(image1250, "build/amer.png")
