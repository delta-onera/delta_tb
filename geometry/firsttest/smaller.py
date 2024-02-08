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
        self.features = tmp.cuda().half()

    def forward(self, x):
        x = x.cuda().half() / 255.0
        x = (x - 0.5) / 0.25
        return self.features(x)


with torch.no_grad():
    net = FeatureExtractor()

    image = PIL.Image.open("/scratchf/DFC2015/BE_ORTHO_27032011_315130_56865.tif")
    image = numpy.asarray(image.convert("RGB").copy())
    image = torch.Tensor(image)[0:9920, 0:9920, :].clone()
    image = torch.stack([image[:, :, 0], image[:, :, 1], image[:, :, 2]], dim=0)
    image = image.unsqueeze(0)

    print("extract data")
    allfeatures = torch.zeros(128, 620, 620)
    for r in range(10):
        for c in range(10):
            x = image[:, :, 992 * r : 992 * (r + 1), 992 * c : 992 * (c + 1)]
            z = net(x)[0].cpu()
            allfeatures[:, 62 * r : 62 * (r + 1), 62 * c : 62 * (c + 1)] = z

    allfeatures = torch.nn.functional.avg_pool2d(
        allfeatures.unsqueeze(0), kernel_size=2, stride=2
    )
    allfeatures = allfeatures[0]

    print("extract stats")
    allfeatures.cuda().half()
    norm = torch.sqrt((allfeatures**2).sum(0).unsqueeze(0)).half()
    allfeatures = allfeatures / (norm + 1)

    allfeatures = allfeatures.flatten(1)
    assert allfeatures.shape == (128, 310 * 310)

    GRAM = torch.matmul(torch.transpose(allfeatures, 0, 1), allfeatures)
    del allfeatures
    torch.fill_diagonal(GRAM, -1)
    assert GRAM.shape == (310 * 310, 310 * 310)

    maxGRAM, _ = GRAM.max(1)
    assert GRAM.shape == (310 * 310)
    del GRAM

    maxGRAM = maxGRAM.view(310, 310)

    image620 = torch.nn.functional.interpolate(image, size=310, mode="bilinear")
    image620 = image620[0] / 255
    torchvision.utils.save_image(image620, "build/image.png")

    image620 *= (maxGRAM < 0).float().unsqueeze(0)
    torchvision.utils.save_image(image620, "build/amer.png")
