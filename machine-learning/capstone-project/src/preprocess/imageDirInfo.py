from os import walk, sep
from PIL import Image
import pandas
import numpy as np


class ImageDirInfo(object):
    "Gives back info of the images in a certain directory"

    def __init__(self, path):
        self.path = path

    def get_filenames(self):
        "Gather the filenames of all files inside this dir"
        filenames = []
        for root, dirs, files in walk(self.path):
            absFiles = [root + sep + filename for filename in files]
            filenames = filenames + absFiles

        return filenames

    def get_size(self, filename):
        "get sizes of all the files inside this dir"

        size = None
        with Image.open(filename) as f:
            size = f.size
        return size

    def get_all_sizes(self):
        "get the frequency table of sizes"
        filenames = self.get_filenames()
        sizes = [self.get_size(filename) for filename in filenames]
        sizes_x, sizes_y = zip(*sizes)
        sizes_x = pandas.Series(sizes_x)
        sizes_y = pandas.Series(sizes_y)

        return sizes_x, sizes_y

    def get_brightness_info(self):
        filenames = self.get_filenames()
        brightnesses = [self.get_brightness(f) for f in filenames]
        (brightness_means, brightness_stds) = zip(*brightnesses)

        brightness_means = np.asarray(brightness_means).astype(float)
        brightness_stds = np.asarray(brightness_stds).astype(float)

        self.plot(pandas.Series(brightness_means))
        plot = self.plot(pandas.Series(brightness_stds))
        plot.legend(['Means', 'Std dev.'], loc='upper right')

    def get_brightness(self, filename):
        img = Image.open(filename)
        img_y, img_b, img_r = img.convert('YCbCr').split()
        img_y = np.asarray(img_y).astype(float)

        img_y /= 255
        return (img_y.mean(), img_y.std())

    def normalize(self, image):
        y, b, r = image.convert('YCbCr').split()
        y_norm = np.asarray(y).astype(float)

        y_norm /= 255
        y_norm -= y_norm.mean()
        y_norm /= y_norm.std()

        scale = np.max([np.abs(np.percentile(y_norm, 1.0)),
                        np.abs(np.percentile(y_norm, 99.0))])
        y_norm /= scale
        y_norm = np.clip(y_norm, -1.0, 1.0)
        y_norm = (y_norm + 1.0) / 2.0

        y_norm = (y_norm * 255 + 0.5).astype(np.uint8)

        y_norm = Image.fromarray(y_norm)

        ybr = Image.merge('YCbCr', (y_norm, b, r))
        rbg = ybr.convert('RGB')

        return rbg

    def resize(self, image, size):
        img_x, img_y = image.size

        if img_x < img_y:
            x_new = int(size * (img_x / img_y) + 0.5)
            y_new = size
        else:
            x_new = size
            y_new = int(size * (img_y / img_x) + 0.5)

        image_resized = image.resize((x_new, y_new), resample=Image.BICUBIC)
        image_padded = Image.new('RGB', (size, size), (128, 128, 128))
        ulc = ((size - x_new) // 2, (size - y_new) // 2)
        image_padded.paste(image_resized, ulc)

        return image_padded

    def plot(self, freqs):
        print(freqs.value_counts())
        return freqs.plot(kind='hist', bins=15)
