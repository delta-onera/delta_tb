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


def debughisto(image):
    tot = image.shape[0] * image.shape[1]
    s = "histo"
    for i in range(256):
        s += "\t" + str(int(numpy.sum(numpy.int32(image == i)) * 1000.0 / tot))
    print(s)


def histogrammatching(image, tmpl):
    # inspired from scikit-image/blob/main/skimage/exposure/histogram_matching.py
    tmpl_quantiles, tmpl_val = tmpl

    _, src_indices, src_counts = numpy.unique(
        image.flatten(), return_inverse=True, return_counts=True
    )

    # ensure single value can not distord the histogram
    cut = numpy.ones(src_counts.shape) * image.shape[0] * image.shape[1] / 20
    src_counts = numpy.minimum(src_counts, cut)
    src_quantiles = numpy.cumsum(src_counts)
    src_quantiles = src_quantiles / src_quantiles[-1]

    interp_a_values = numpy.interp(src_quantiles, tmpl_quantiles, tmpl_val)
    return interp_a_values[src_indices].reshape(image.shape)


class ManyHistogram:
    def __init__(self):
        self.cibles = numpy.zeros((5, 256))

        centers = [256 // 3, 256 // 2, 256 * 2 // 3]
        for c in range(3):
            for i in range(256):
                self.cibles[c][i] = 15.0 * math.exp(-((centers[c] - i) ** 2) / 255) + 1

        for i in range(256):
            self.cibles[3][i] = 256 // 2 - abs(i - 256 // 2) + 2

        self.cibles[4] = numpy.ones(256)

        self.cumsum = []
        for i in range(5):
            vals = numpy.arange(256)
            quantiles = numpy.cumsum(self.cibles[i])
            quantiles = quantiles / quantiles[-1]
            self.cumsum.append((quantiles, vals))

    def normalize(self, image):
        image = numpy.int32(image)
        out = numpy.zeros((18, image.shape[0], image.shape[1]))
        out = numpy.int16(out)

        for i in range(5):
            for ch in range(3):
                out[i * 3 + ch] = histogrammatching(image[:, :, ch], self.cumsum[i])

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

    for i in range(6):
        debug = images[3 * i : 3 * i + 3, :, :]
        debug = numpy.uint8(numpy.transpose(debug, axes=(1, 2, 0)))
        debug = PIL.Image.fromarray(debug)
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
