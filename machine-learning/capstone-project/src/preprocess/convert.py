from __future__ import print_function
from PIL import Image
from imageDirInfo import ImageDirInfo
import os


def convert(from_dir, to_dir, size):
    if os.path.exists(to_dir):
        print("To directory already exists, make sure it's a new directory")
        return

    os.makedirs(to_dir)

    idi = ImageDirInfo(from_dir)
    filenames = idi.get_filenames()

    for i, filename in enumerate(filenames):
        if i % 10 == 0:
            print('{} / {} converted.'.format(i, len(filenames)), end='\r')
        image = Image.open(filename)
        normalized = idi.normalize(image)
        padded = idi.resize(normalized, size)
        padded.save(to_dir + '/' + os.path.basename(filename))
    print('Done                  ')
