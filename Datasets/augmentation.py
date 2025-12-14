import numpy as np
import torch
from PIL import Image

class TransResize():
    def __init__(self, fine_size = 286):
        # super(RandomFlip, self).__init__()
        self.fine_size = fine_size

    def __call__(self, image, label):

        h, w, c = image.shape
        min_len = np.min([h, w])
        
        image = np.asarray(Image.fromarray(image).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        label = np.asarray(Image.fromarray(label).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        return image, label

class TransCrop():
    def __init__(self, crop_size=256):
        # super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    def __call__(self, image, label):
        w, h, c = image.shape

        h1 = np.random.randint(0, h - self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)

        image = image[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]
        label = label[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]

        return image, label

class TransFlip():
    def __init__(self, prob=0.5):
        # super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label

class TransResize3():
    def __init__(self, fine_size = 286):
        # super(RandomFlip, self).__init__()
        self.fine_size = fine_size

    def __call__(self, image, label, mask):

        h, w, c = image.shape
        min_len = np.min([h, w])
        
        image = np.asarray(Image.fromarray(image).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        label = np.asarray(Image.fromarray(label).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.BICUBIC), dtype=np.int64)
        mask = np.asarray(Image.fromarray(mask).resize((w * self.fine_size / min_len, h * self.fine_size / min_len), Image.NEAREST), dtype=np.int64)
        return image, label, mask

class TransCrop3():
    def __init__(self, crop_size=256):
        # super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    def __call__(self, image, label, mask):
        w, h, c = image.shape

        h1 = np.random.randint(0, h - self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)

        image = image[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]
        label = label[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]
        mask = mask[w1 : (w1 + self.crop_size), h1 : (h1 + self.crop_size)]


        return image, label, mask

class TransFlip3():
    def __init__(self, prob=0.5):
        # super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label, mask):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
            mask = mask[:,::-1]
        return image, label, mask


class TransResizeN_:
    def __init__(self, fine_size=286):
        # super(RandomFlip, self).__init__()
        self.fine_size = fine_size

    def __call__(self, *args):
        h, w, c = args[0].shape
        min_len = np.min([h, w])
        res = []
        for arg in args:
            res.append(np.asarray(
            Image.fromarray(arg).resize((int(w * self.fine_size / min_len), int(h * self.fine_size / min_len)), Image.BICUBIC),
            dtype=np.int64))
        return res


class TransCropN_:
    def __init__(self, crop_size=256):
        # super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    def __call__(self, *args):
        w, h, c = args[0].shape

        h1 = np.random.randint(0, h - self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)

        res = []
        for arg in args:
            res.append(arg[w1: (w1 + self.crop_size), h1: (h1 + self.crop_size)])

        return res


class TransFlipN_:
    def __init__(self, prob=0.5):
        # super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, *args):
        res = []
        if np.random.rand() < self.prob:
            for arg in args:
                res.append(arg[:, ::-1])
            return res
        else:
            return args

class TransResizeN:
    def __init__(self, fine_size=286):
        # super(RandomFlip, self).__init__()
        self.fine_size = fine_size
        self.size = int(self.fine_size), int(self.fine_size * 1.125)

    def __call__(self, *args):
        # b, c, h, w = args[0].shape
        # min_len = np.min([h, w])
        res = []
        for arg in args:
            res.append(arg.resize(self.size, mode='nearest' if arg.dtype == torch.uint8 else 'bilinear'))
        return res


class TransCropN:
    def __init__(self, crop_size=256):
        # super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    def __call__(self, *args):
        b, c, h, w = args[0].shape

        h1 = np.random.randint(0, h - self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)

        res = []
        for arg in args:
            res.append(arg.crop((w1, w1 + self.crop_size, h1, h1 + self.crop_size), mode='xxyy'))

        return res


class TransFlipN:
    def __init__(self, prob=0.5):
        # super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, *args):
        res = []
        if np.random.rand() < self.prob:
            for arg in args:
                res.append(arg.hflip())
            return res
        else:
            return args
