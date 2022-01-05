import numpy
import math


def minmax(image, removeborder=True):
    values = list(image.flatten())
    if removeborder:
        values = sorted(values)
        I = len(values)
        values = values[(I * 3) // 100 : (I * 97) // 100]
        imin = values[0]
        imax = values[-1]
    else:
        imin = min(values)
        imax = max(values)

    if imin == imax:
        return numpy.int16(256 // 2 * numpy.ones(image.shape))

    out = 255.0 * (image - imin) / (imax - imin)
    out = numpy.int16(out)

    tmp = numpy.int16(out >= 255)
    out -= 10000 * tmp
    out *= numpy.int16(out > 0)
    out += 255 * tmp

    return out


def printhisto(histo):
    s = "histo"
    for i in histo:
        histo[i] = int(1000 * histo[i])
        s += "\t" + str(histo[i])
    print(s)
    return histo


def computehisto(image, removelarge=True):
    keys = set(image.flatten())
    source = {}
    for k in keys:
        source[k] = numpy.sum(numpy.int32(image == k))

    sourcesum = 1.0 / image.shape[0] / image.shape[1]
    for k in keys:
        source[k] *= sourcesum
    return source


def histogrammatching(source, cible):
    j = 0
    matching = {}
    for i in source:
        matching[i] = j
        cible[j] -= source[i]
        if cible[j] < 0.0:
            j += 1
            if j > 255:
                j = 255

    for i in matching:
        matching[i] = int(matching[i])

    inversematching = {}
    for i in matching:
        inversematching[matching[i]] = int(i)

    for i in range(256):
        if i not in inversematching:
            inversematching[i] = inversematching[i - 1]
    return inversematching


def convert(image, matching):
    output = numpy.int16(numpy.zeros(image.shape))
    for i in range(255):
        output += numpy.int16(image > matching[i + 1])
    return minmax(output, removeborder=False)


class ManyHistogram:
    def __init__(self):
        self.cibles = numpy.zeros((5, 256))

        centers = [256 // 3, 256 // 2, 256 * 2 // 3]
        for c in range(3):
            for i in range(256):
                self.cibles[c][i] += 10.0 * math.exp(-((centers[c] - i) ** 2) / 255)

        for i in range(256):
            self.cibles[3][i] = 256 // 2 - abs(i - 256 // 2) + 2

        self.cibles[4] = numpy.ones(256)

        self.cibles = numpy.float32(self.cibles)
        for i in range(5):
            self.cibles[i] = self.cibles[i] / numpy.sum(self.cibles[i])

    def normalize(self, image):
        image = numpy.int32(image)
        out = numpy.zeros((18, image.shape[0], image.shape[1]))
        out = numpy.int16(out)

        source = [computehisto(image[:, :, i]) for i in range(3)]
        for i in range(5):
            for ch in range(3):
                tmp = histogrammatching(source[ch], self.cibles[i].copy())
                out[i * 3 + ch] = convert(image[:, :, ch], tmp)

        out[15] = minmax(image[:, :, 0])
        out[16] = minmax(image[:, :, 1])
        out[17] = minmax(image[:, :, 2])

        return out


if __name__ == "__main__":
    normalizations = ManyHistogram()

    import PIL
    from PIL import Image

    image = PIL.Image.open("/data/miniworld/bruges/train/1_x.png").convert("RGB").copy()
    image = numpy.uint8(numpy.asarray(image))

    images = normalizations.normalize(image)

    printhisto(computehisto(image[:, :, 0]))
    printhisto(computehisto(images[:, :, 0]))
    quit()

    for i in range(6):
        debug = images[3 * i : 3 * i + 3, :, :]
        debug = numpy.transpose(debug, axes=(1, 2, 0))
        debug = PIL.Image.fromarray(numpy.uint8(debug))
        debug.save("build/test8_" + str(i) + ".png")

    import rasterio

    with rasterio.open("/data/SEMCITY_TOULOUSE/TLS_BDSD_M_04.tif") as src:
        r = numpy.uint16(src.read(4))
        g = numpy.uint16(src.read(3))
        b = numpy.uint16(src.read(2))
        image16bits = numpy.stack([r, g, b], axis=-1)

    images8bits = normalizations.normalize(image16bits)
    for i in range(6):
        debug = images8bits[3 * i : 3 * i + 3, :, :]
        debug = numpy.transpose(debug, axes=(1, 2, 0))
        debug = PIL.Image.fromarray(numpy.uint8(debug))
        debug.save("build/test16_" + str(i) + ".png")
