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


def extractKeyCells(x, nb=300):
    assert len(x.shape) == 3

    x = x / torch.sqrt((x * x + 0.1).sum(0).unsqueeze(0))
    x = x.cuda().flatten(1)

    GRAM = torch.matmul(x.transpose(0, 1), x)
    GRAM = torch.nn.functional.softmax(GRAM, 1)
    GRAM = torch.diagonal(GRAM)

    seuil = list(GRAM.cpu().numpy())
    seuil = float(sorted(seuil)[-nb])
    GRAM = GRAM.view(x.shape[1], x.shape[2])

    return GRAM >= seuil


with torch.no_grad():
    net = FeatureExtractor()

    image = PIL.Image.open("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")
    image = numpy.asarray(image.convert("RGB").copy())
    image = torch.Tensor(image)[0:9600, 0:9600, :].clone()
    image = torch.stack([image[:, :, 0], image[:, :, 1], image[:, :, 2]], dim=0)
    image = image.unsqueeze(0)

    print("extract data")
    allfeatures = torch.zeros(256, 150, 150)
    for r in range(10):
        for c in range(10):
            x = image[:, :, 960 * r : 960 * (r + 1), 960 * c : 960 * (c + 1)]
            z = net(x)[0].cpu()
            allfeatures[:, 15 * r : 15 * (r + 1), 15 * c : 15 * (c + 1)] = z

    allfeatures = allfeatures / torch.sqrt(
        (allfeatures * allfeatures + 0.1).sum(0).unsqueeze(0)
    )

    print("extract stats")
    allfeatures = allfeatures.cuda().flatten(1)

    GRAM = torch.matmul(allfeatures.transpose(0, 1), allfeatures)
    del allfeatures
    GRAM = torch.nn.functional.softmax(GRAM, 1)
    GRAM = torch.diagonal(GRAM)
    assert GRAM.shape[0] == (150 * 150)

    seuil = list(GRAM.cpu().numpy())
    seuil = float(sorted(seuil)[-300])
    print(seuil)
    GRAM = GRAM.view(150, 150).cpu()
    print((GRAM >= seuil).float().sum())

    imagetiny = torch.nn.functional.interpolate(image, size=150, mode="bilinear")
    imagetiny = imagetiny[0] / 255
    torchvision.utils.save_image(imagetiny, "build/image.png")

    imagetiny *= (GRAM >= seuil).float().unsqueeze(0)
    torchvision.utils.save_image(imagetiny, "build/amer.png")
