import random
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from torch import nn

from cor import UNI, RANDOM, SIZE


class ImageAugment(nn.Module):
    def __init__(self, style_idx, shift_idx):
        """
        ImageAugment
        :param style_idx: the index of style augment
        :param shift_idx: the index of position augment
        """
        super(ImageAugment, self).__init__()
        self.style_idx = style_idx
        self.shift_idx = shift_idx

        if style_idx == "bright":
            self.style = iaa.imgcorruptlike.Brightness()
        elif style_idx == "rain":
            self.style = iaa.Rain()
        elif style_idx == "snow":
            self.style = iaa.Snowflakes()
        elif style_idx == "fog":
            self.style = iaa.Fog()
        elif style_idx == "cutout":
            self.style = iaa.Cutout(size=SIZE)
        else:
            self.style = None

        if self.shift_idx == "random":
            self.random_lat = RANDOM * 9e-6
            self.random_lon = RANDOM * 1.043e-5
            self.hei = False
        elif self.shift_idx == "uni":
            self.lat_aug = None
            self.lon_aug = None
            self.random_lat = UNI * 9e-6
            self.random_lon = UNI * 1.043e-5
            self.hei = False
        elif self.shift_idx == "hei":
            self.random_lat = 0
            self.random_lon = 0
            self.hei = True
        else:
            self.random_lat = 0
            self.random_lon = 0
            self.hei = False

    def forward(self, image):
        """
        forward pass of ImageAugment
        :param image: the provided input image
        :return: the image after augmented
        """
        if self.style is not None:
            image = self.style(image=image)
        return image

    def forward_shift(self):
        """
        forward pass of position augment
        :return: the position shift after augmented
        """
        if self.shift_idx == "random":
            random_lat_shift = random.uniform(-self.random_lat, self.random_lat)
            random_lon_shift = random.uniform(-self.random_lon, self.random_lon)
        elif self.shift_idx == "uni":
            if self.lat_aug is None:
                random_lat_shift = random.uniform(-self.random_lat, self.random_lat)
                random_lon_shift = random.uniform(-self.random_lon, self.random_lon)
                self.lat_aug = 1 if random_lat_shift >= 0 else -1
                self.lon_aug = 1 if random_lon_shift >= 0 else -1
            else:
                if self.lat_aug == 1:
                    random_lat_shift = random.uniform(0, self.random_lat)
                else:
                    random_lat_shift = random.uniform(-self.random_lat, 0)
                if self.lon_aug == 1:
                    random_lon_shift = random.uniform(0, self.random_lon)
                else:
                    random_lon_shift = random.uniform(-self.random_lon, 0)
        else:
            random_lat_shift = 0
            random_lon_shift = 0
        return [random_lat_shift, random_lon_shift]


if __name__ == '__main__':
    image_augment = ImageAugment(style_idx="bright", shift_idx="ori")
    pic = Image.open("../1.png")

    pic = np.array(pic)
    pic = image_augment(pic)
    pic = Image.fromarray(pic)
    pic.save("../2.png")
