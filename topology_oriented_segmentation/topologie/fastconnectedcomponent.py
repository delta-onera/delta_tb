import torch


def maxpool(y):
    return torch.nn.functional.max_pool2d(y, kernel_size=3, stride=1, padding=1)


def minpool(y):
    yy = 1 - y
    yyy = maxpool(yy)
    return 1 - yyy


def isole(y):
    globalresize = torch.nn.AdaptiveMaxPool2d((y.shape[0], y.shape[1]))
    yy = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)
    yyy = torch.nn.functional.avg_pool2d(
        yy, kernel_size=3, stride=1, padding=1, divisor_override=1
    )
    alone = (yy == 1).float() * (yyy == 1).float()
    return globalresize(alone)


def connected_component_seed(y):
    with torch.no_grad():
        if len(y.shape) == 2:
            adddim = True
            y = y.clone().unsqueeze(0).float()

        erode = y.clone()
        for i in range(15):
            erode = (minpool(y) + isole(y) >= 1).float()
        return erode


def extract_critical_background(y):
    CCseeds = connected_component_seed(y)

    with torch.no_grad():
        tmp = torch.range(y.shape[0] * y.shape[1] * y.shape[2])
        tmp = tmp.view(y.shape)

        ConnecComp1 = CCseeds * tmp
        ConnecComp2 = CCseeds * (y.shape[0] * y.shape[1] * y.shape[2] - tmp)
        for i in range(15):
            ConnecComp1 = maxpool(ConnecComp1) * (y == 1).float()
            ConnecComp2 = maxpool(ConnecComp2) * (y == 1).float()

        for i in range(3):
            ConnecComp1 = maxpool(ConnecComp1)
            ConnecComp2 = maxpool(ConnecComp2)
        return (y == 0).float() * (ConnecComp1 != ConnecComp2).float()
