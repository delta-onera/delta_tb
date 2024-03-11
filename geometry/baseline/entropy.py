import torch
import torchvision
import generate


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

        x1 = self.features(x)
        if self.naif:
            return x1

        tmp = torch.rot90(x, k=1, dims=[2, 3])
        x2 = torch.rot90(self.features(tmp), k=-1, dims=[2, 3])

        tmp = torch.rot90(x, k=2, dims=[2, 3])
        x3 = torch.rot90(self.features(tmp), k=-2, dims=[2, 3])

        tmp = torch.rot90(x, k=3, dims=[2, 3])
        x4 = torch.rot90(self.features(tmp), k=-3, dims=[2, 3])

        xM = torch.maximun(torch.maximum(x1, x2), torch.maximum(x3, x4))
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

        GRAM = torch.matmul(allfeatures.transpose(0, 1), allfeatures)
        del allfeatures
        GRAM = torch.nn.functional.softmax(GRAM, 1)
        GRAM = torch.diagonal(GRAM)

        GRAM.sort()
        _, pos = torch.topk(GRAM, k, dim=0)

        posR = (pos / h).long()
        posW = pos % w
        posR, posW = 8 * posR + 4, 8 * posW + 4
        return posR, posW


with torch.no_grad():
    net = FeatureExtractor().cuda()
    data = generate.Generator()

    for i in range(10):
        x, xx, proj = data.get()
        P = extractSalientPoint(xx, 4)

    imagetiny = torch.nn.functional.interpolate(image, size=150, mode="bilinear")
    imagetiny = imagetiny[0] / 255
    torchvision.utils.save_image(imagetiny, "build/image.png")

    imagetiny *= (GRAM >= seuil).float().unsqueeze(0)
    torchvision.utils.save_image(imagetiny, "build/amer.png")
