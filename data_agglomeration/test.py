import os
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

whereIam = os.uname()[1]
assert whereIam in ["super", "wdtis719z", "ldtis706z"]


print("load model")
with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()

import dataloader

print("massif benchmark")
cm = {}
with torch.no_grad():
    if whereIam in ["super", "wdtis719z"]:
        availabledata = ["toulouse", "potsdam"]
        root = "/data/miniworld/"

    if whereIam in ["ldtis706z"]:
        availabledata = ["toulouse", "potsdam", "bruges", "newzealand"]
        root = "/media/achanhon/bigdata/data/miniworld/"

    for name in availabledata:
        data = dataloader.SegSemDataset(
            root + name + "/test",
            name in ["toulouse", "potsdam", "bruges", "newzealand"],
        )

        cm[name] = np.zeros((2, 2), dtype=int)
        for i in range(data.nbImages):
            imageraw, label = data.getImageAndLabel(i)

            image = torch.Tensor(np.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            globalresize = torch.nn.AdaptiveAvgPool2d((image.shape[2], image.shape[3]))
            power2resize = torch.nn.AdaptiveAvgPool2d(
                (data.shape[2] // 32) * 32, (data.shape[3] // 32) * 32
            )
            image = power2resize(image)

            if image.shape[2] > 512 or image.shape[3] > 512:
                tilesize, stride = 128, 32
                pred = torch.zeros(2, image.shape[2], image.shape[3]).cpu()
                for row in range(0, image.shape[2] - tilesize + 1, stride):
                    for col in range(0, image.shape[3] - tilesize + 1, stride):
                        tmp = net(
                            image[:, :, row : row + tilesize, col : col + tilesize]
                            .float()
                            .to(device)
                        ).cpu()
                        pred[:, row : row + tilesize, col : col + tilesize] += tmp[0]
            else:
                pred = net(image.to(device))

            pred = globalresize(pred)
            _, pred = torch.max(pred[0], 0)
            pred = pred.cpu().numpy()
            assert label.shape == pred.shape

            cm[name] += confusion_matrix(
                label.flatten(), pred.flatten(), list(range(nbclasses))
            )

            if name in ["toulouse", "potsdam"]:
                imageraw = PIL.Image.fromarray(np.uint8(imageraw))
                pred.save("build/" + name + "_" + str(i) + "_x.png")
                label = PIL.Image.fromarray(np.uint8(label) * 255)
                pred.save("build/" + name + "_" + str(i) + "_y.png")
                pred = PIL.Image.fromarray(np.uint8(pred) * 255)
                pred.save("build/" + name + "_" + str(i) + "_z.png")

        print(
            name,
            cm[name][0][0],
            cm[name][0][1],
            cm[name][1][0],
            cm[name][1][1],
            100.0 * (cm[name][0][0] + cm[name][1][1]) / np.sum(cm[name]),
            50.0 * cm[name][0][0] / (cm[name][0][0] + cm[name][1][0] + cm[name][0][1])
            + 50.0
            * cm[name][1][1]
            / (cm[name][1][1] + cm[name][1][0] + cm[name][0][1]),
        )

print("summary")
for name in availabledata:
    print(
        name,
        50.0 * cm[name][0][0] / (cm[name][0][0] + cm[name][1][0] + cm[name][0][1])
        + 50.0 * cm[name][1][1] / (cm[name][1][1] + cm[name][1][0] + cm[name][0][1]),
    )
