import numpy
import math


def minmax(image):
    values = list(image.fold())
    sorted(values)
    I = len(values)
    values = values[(I*3)//100:(I*97)//100]
    imin = values[0]
    imax = values[-1]
    
    out = 255.*(image - imin)/(imax-imin+1)
    out = numpy.int32(out)
    
    tmp = numpy.int32(out>255)
    out -= 100000*tmp
    out *= numpy.int32(out>0)
    out += 255*tmp
    
    return out


def computehisto(image):
    keys = set(image.fold())
    source = {}
    for k in keys:
        source[k] = numpy.sum(numpy.int32(image==k))
    return source
    

def histogrammatching(source,cible):
    #assert |source|>>|cible|
    cible, source = numpy.float64(cible), numpy.float64(source)
    
    I = numpy.non_zero(source>numpy.sum(source)/255)
    for i in I:
        source[i]=numpy.sum(source)/255
    cible, source = cible/numpy.sum(cible), source/numpy.sum(source)
    
    j=0
    matching={}
    for i in range(source.shape[0]):
        matching[i]=j
        cible[j]-=source[i]
        if cible[j]<=0:
            j+=1
            if j>255:
                j=255
    if j<255:
        for i in matching:
            matching[i]*=(255/j)
    
    inversematching = {}
    for i in matching:
        inversematching[matching[i]]=int(i)
    return inversematching
    
def convert(image, matching):
    output = numpy.zeros(image.shape)
    for i in range(255):
        outputs +=numpy.int32(image>matching[i+1])
    return output
    


    

class ManyHistogram:
    def __init__(self):
        self.cibles = numpy.zeros(5,256)
        
        centers = [256//3,256//2,256*2//3]
        for c in range(3):
            for i in range(256):
                self.cibles[c][i] += 10.*math.exp(-(centers[c] - i)**2/255)
                
        for i in range(256):
            self.cibles[3][i] = 256//2-abs(i-256//2)+2
        self.cibles[4]= numpy.ones(256)


    def normalize(self, image):
        image = numpy.int32(image)
        out = numpy.zeros(18,image.shape[0],image.shape[1])
        
        source = [computehisto(image[:,:,i]) for i in range(3)]
        for i in range(5):
            for ch in range(3):
                tmp = histogrammatching(source[ch],self.cibles[i])
                out[i*3+ch] = convert(image[:,:,ch],tmp)
        
        out[15] = minmax(image[:,:,0])
        out[16] = minmax(image[:,:,1])
        out[17] = minmax(image[:,:,2])
        
        return out


if __name__=="__main__":
    normalizations = ManyHistogram()
    
    import PIL
    from PIL import Image
    image = PIL.Image.open("/data/miniworld/bruges/train/1_x.png").convert("RGB").copy()
    image = numpy.uint8(numpy.asarray(image))
    
    images = normalizations.normalize(image)
    for i in range(6):
        debug = images[3*i:3*i+3,:,:]
        debug = numpy.transpose(debug, axes=(1, 2, 0))
        debug = PIL.Image.fromarray(numpy.uint8(debug))
        debug.save("build/test8_"+str(i)+".png")
    
    
    import rasterio
    with rasterio.open("/data/SEMCITY_TOULOUSE/TLS_BDSD_M_04.tif") as src:
        image16bits = numpy.uint16(src.read(1))
        
    images8bits = normalizations.normalize(image16bits)
    for i in range(6):
        debug = images8bits[3*i:3*i+3,:,:]
        debug = numpy.transpose(debug, axes=(1, 2, 0))
        debug = PIL.Image.fromarray(numpy.uint8(debug))
        debug.save("build/test16_"+str(i)+".png")
            
