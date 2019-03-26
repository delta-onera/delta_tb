import numpy as np
import random
import numbers
import scipy.ndimage as ndimage

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target, mask=None):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
            if type(target) == list:
                target = [np.copy(np.fliplr(gt)) for gt in target]
                for gt in target:
                    gt[:,:,0] *= -1
            else:
                target = np.copy(np.fliplr(target))
                target[:,:,0] *= -1
            if not mask is None:
                if type(mask) == list:
                    mask = [np.copy(np.fliplr(m)) for m in mask]
                else:
                    mask = np.copy(np.fliplr(mask))

        if mask is None:
            return inputs, target
        else:
            return inputs, target, mask


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target, mask=None):
        if random.random() < 0.5:
            inputs = [np.copy(np.flipud(im)) for im in inputs]
            if type(target) == list:
                target = [np.copy(np.flipud(gt)) for gt in target]
                for gt in target:
                    gt[:,:,1] *= -1
            else:
                target = np.copy(np.flipud(target))
                target[:,:,1] *= -1
            if not mask is None:
                if type(mask) == list:
                    mask = [np.copy(np.flipud(m)) for m in mask]
                else:
                    mask = np.copy(np.flipud(mask))

        if mask is None:
            return inputs, target
        else:
            return inputs, target, mask

class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs,target):
        if type(target) == list:
            raise TypeError('Multiframe non géré avec RandomRotate, target ne peut pas être une liste')

        applied_angle = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180

        h, w, _ = target.shape

        def rotate_flow(i,j,k):
            return -k*(j-w/2)*(diff*np.pi/180) + (1-k)*(i-h/2)*(diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] + np.sin(angle1_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target_[:,:,1]
        return inputs,target

class RandomRotateSimple(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    Pas d'ajout de mouvement interimage, les deux images prennent la même translation
    """

    def __init__(self, angle, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, inputs, target, mask=None):

        applied_angle = random.uniform(-self.angle,self.angle)
        angle_rad = applied_angle*np.pi/180

        if type(target) == list:
            h, w, _ = target[0].shape
        else:
            h, w, _ = target.shape

        inputs = [ndimage.interpolation.rotate(im, applied_angle, reshape=self.reshape,
                                                order=self.order) for im in inputs]
        if type(target) == list:
            target = [ndimage.interpolation.rotate(gt, applied_angle,
                            reshape=self.reshape, order=self.order) for gt in target]
            for gt in target:
                # flow vectors must be rotated too! careful about Y flow which is upside down
                gt_ = np.copy(gt)
                gt[:,:,0] = np.cos(angle_rad)*gt_[:,:,0] + np.sin(angle_rad)*gt_[:,:,1]
                gt[:,:,1] = -np.sin(angle_rad)*gt_[:,:,0] + np.cos(angle_rad)*gt_[:,:,1]
        else:
            target = ndimage.interpolation.rotate(target, applied_angle,
                                        reshape=self.reshape, order=self.order)
            # flow vectors must be rotated too! careful about Y flow which is upside down
            target_ = np.copy(target)
            target[:,:,0] = np.cos(angle_rad)*target_[:,:,0] + np.sin(angle_rad)*target_[:,:,1]
            target[:,:,1] = -np.sin(angle_rad)*target_[:,:,0] + np.cos(angle_rad)*target_[:,:,1]

        if mask is None:
            return inputs, target
        else:
            if type(mask) == list:
                mask = [ndimage.interpolation.rotate(m, applied_angle,
                                reshape=self.reshape, order=self.order) for m in mask]
            else:
                mask = ndimage.interpolation.rotate(mask, applied_angle,
                                            reshape=self.reshape, order=self.order)
            return inputs, target, mask

class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        if type(target) == list:
            raise TypeError('Multiframe non géré avec RandomRotate, target ne peut pas être une liste')

        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target
