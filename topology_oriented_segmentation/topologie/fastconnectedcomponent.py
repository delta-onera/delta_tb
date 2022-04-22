import torch


def maxpool(y):
    return torch.nn.functional.max_pool2d(y, kernel_size=3, stride=1, padding=1)


def minpool(y):
    yy = 1 - y
    yyy = maxpool(yy)
    return 1 - yyy


def isborder(y):
    y0, y1 = (y == 0).float(), (y == 1).float()
    y00, y11 = maxpool(y0), maxpool(y1)
    y00, y11 = maxpool(y00), maxpool(y11)
    border = (y1 * y00 + y0 * y11) > 0
    return border.float()


def extract_critical_background(y):
    with torch.no_grad():
        S = y.shape[0] * y.shape[1] * y.shape[2]
        tmp = torch.range(1, S)
        tmp = tmp.view(y.shape)

        ConnecComp = y * tmp
        for i in range(20):
            ConnecComp = maxpool(ConnecComp) * (y == 1).float()

        ConnecComp1, ConnecComp2 = ConnecComp.clone(), S - ConnecComp.clone()
        for i in range(3):
            ConnecComp1 = maxpool(ConnecComp1)
            ConnecComp2 = maxpool(ConnecComp2)
        ConnecComp2 = S - ConnecComp2

        return (y == 0).float() * (ConnecComp1 != ConnecComp2).float()


if __name__ == "__main__":
    import PIL
    from PIL import Image
    import numpy
    import torchvision

    label = PIL.Image.open("/data/miniworld/potsdam/train/6_y.png").convert("L").copy()
    label = numpy.uint8(numpy.asarray(label))
    label = numpy.uint8(label != 0)

    y = torch.Tensor(label)
    y = y.unsqueeze(0).float()

    torchvision.utils.save_image(y, "build/raw.png")
    torchvision.utils.save_image(minpool(y), "build/minpool.png")
    torchvision.utils.save_image(maxpool(y), "build/maxpool.png")

    border = isborder(y)
    border = y * (1 - border) + 0.5 * border

    torchvision.utils.save_image(border, "build/border.png")

    criticalback = extract_critical_background(y)
    torchvision.utils.save_image(criticalback, "build/critical.png")
