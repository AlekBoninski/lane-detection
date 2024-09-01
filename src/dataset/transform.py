import random

from PIL import Image, ImageOps
from PIL.Image import Transpose
from torchvision.transforms import ToTensor, Resize, InterpolationMode


class EncoderResize:
    def __init__(self, height=512):
        self.height = height

    def __call__(self, image, mask):
        image = Resize((self.height, self.height * 2), InterpolationMode.BILINEAR)(image)
        mask = Resize((self.height, self.height * 2), InterpolationMode.NEAREST)(mask)
        mask = Resize(int(self.height / 8), InterpolationMode.NEAREST)(mask)

        return image, mask


class ERFNetResize:
    def __init__(self, height=512):
        self.height = height

    def __call__(self, image, mask):
        image = Resize((self.height, self.height * 2), InterpolationMode.BILINEAR)(image)
        mask = Resize((self.height, self.height * 2), InterpolationMode.NEAREST)(mask)

        return image, mask


class ImageAndMaskToTensor:
    def __call__(self, image, mask):
        image = ToTensor()(image)
        mask = ToTensor()(mask)

        return image, mask.long()


class MaskRelabel:
    def __call__(self, image, mask):
        mask[mask == 255] = 1

        return image, mask


class RandomAugment:
    def __call__(self, image, mask):
        #  Flip horizontally with 50% chance
        should_flip = random.random()
        if should_flip < 0.5:
            image = image.transpose(Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Transpose.FLIP_LEFT_RIGHT)

        #  Random translations by 2 pixels
        translate_x = random.randint(-2, 2)
        translate_y = random.randint(-2, 2)

        image = ImageOps.expand(image, border=(translate_x, translate_y, 0, 0), fill=0)
        image = image.crop((0, 0, image.size[0] - translate_x, image.size[1] - translate_y))
        mask = ImageOps.expand(mask, border=(translate_x, translate_y, 0, 0), fill=0)
        mask = mask.crop((0, 0, mask.size[0] - translate_x, mask.size[1] - translate_y))

        return image, mask
