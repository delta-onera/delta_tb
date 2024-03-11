import torch
import torchvision
import generate
import PIL
from PIL import Image
import numpy


class FeatureExtractor(torch.nn.Module):
    def __init__(self, naif=False):
        super(FeatureExtractor, self).__init__()
        tmp = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features
        """
        3,256,256 -> 1280, 8, 8
        del tmp[7] -> 256, 8, 8
        del tmp[6] -> 160,16,16
        del tmp[5] -> 128,16,16
        del tmp[4] ->  64,32,32
        """
        del tmp[7]
        del tmp[6]
        del tmp[5]
        del tmp[4]
        self.features = tmp.cuda()
        self.naif = naif

    def forward(self, x):
        x = x.cuda() / 255.0
        x = (x - 0.5) / 0.25
        print(x.shape)
        x = torch.nn.functional.pad(x, [32] * 4, mode="reflect")
        print(x.shape)

        x1 = self.features(x)[:, :, 4:-4, 4:-4]
        print(x1.shape)
        if self.naif:
            return x1

        tmp = torch.rot90(x, k=1, dims=[2, 3])
        x2 = torch.rot90(self.features(tmp), k=-1, dims=[2, 3])

        tmp = torch.rot90(x, k=2, dims=[2, 3])
        x3 = torch.rot90(self.features(tmp), k=-2, dims=[2, 3])

        tmp = torch.rot90(x, k=3, dims=[2, 3])
        x4 = torch.rot90(self.features(tmp), k=-3, dims=[2, 3])

        x2, x3, x4 = x2[:, :, 4:-4, 4:-4], x3[:, :, 4:-4, 4:-4], x4[:, :, 4:-4, 4:-4]

        xM = torch.max(torch.max(x1, x2), torch.max(x3, x4))
        xm = 0.25 * (x1 + x2 + x3 + x4)
        return torch.cat([xM, xm], dim=1)


def extractSalientPoint(x, k):
    with torch.no_grad():
        allfeatures = net(x.unsqueeze(0).cuda())[0]
        _, h, w = allfeatures.shape

        allfeatures = allfeatures.flatten(1)
        allfeatures = allfeatures / torch.sqrt(
            (allfeatures * allfeatures + 0.1).sum(0).unsqueeze(0)
        )
        allfeatures[:, 0, :] = 0
        allfeatures[:, -1, :] = 0
        allfeatures[:, :, 0] = 0
        allfeatures[:, :, -1] = 0

        GRAM = torch.matmul(allfeatures.transpose(0, 1), allfeatures)
        del allfeatures
        GRAM = torch.nn.functional.softmax(GRAM, 1)
        GRAM = torch.diagonal(GRAM)

        GRAM.sort()
        _, pos = torch.topk(GRAM, k, dim=0)

        posH = (pos / h).long()
        posW = pos % w
        posH, posW = 8 * posH + 4, 8 * posW + 4
        return posH, posW


def drawRect(path, x, posH, posW, posH_=None, posW_=None):
    x = torch.stack([x[0], x[1], x[2]], dim=-1)
    x = numpy.uint8((x.cpu().numpy() * 255.9))
    image = PIL.Image.fromarray(x)

    draw = PIL.ImageDraw.Draw(image)
    for i in range(posW.shape[0]):
        rect = [posW[i] - 8, posH[i] - 8, posW[i] + 8, posH[i] + 8]
        draw.rectangle(rect, outline="green", width=2)

    if posH_ is None:
        image.save(path)
        return

    for i in range(posW_.shape[0]):
        rect = [posW_[i] - 8, posH_[i] - 8, posW_[i] + 8, posH_[i] + 8]
        draw.rectangle(rect, outline="red", width=2)
    image.save(path)


with torch.no_grad():
    net = FeatureExtractor().cuda()
    data = generate.Generator()

    for i in range(10):
        x, xx, proj = data.get()

        posH, posW = extractSalientPoint(xx, 4)
        drawRect("build/" + str(i) + "_s.png", xx, posH, posW)

        posH_, posW_ = data.oldCoordinate(posH, posW, proj)
        posH, posW = extractSalientPoint(x, 64)
        drawRect("build/" + str(i) + "_b.png", x, posH, posW, posH_, posW_)
