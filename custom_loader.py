import glob
import os
from os.path import join

import numpy as np
from matplotlib import image as mpimg, pyplot as plt


class ClassificationLoader:
    x_dim = 64
    y_dim = 128

    def __init__(self, path, crops_per_negative=3, include_mirrors=False):
        positives_path = join(path, 'positives')
        negatives_path = join(path, 'negatives')

        x = []
        y = []

        for filepath in glob.iglob(f'{path}/negatives/*.png'):
            print(filepath)


def create_classification_dataset(path,
                                  crops_per_negative=3,
                                  include_mirrors=False):

    def extract_patch(image, x_d=64, y_d=128):
        x = np.random.randint(0, image.shape[1] - x_d)
        y = np.random.randint(0, image.shape[0] - y_d)
        patch = image[y: y + y_d, x: x + x_d]

        return patch

    x = []
    y = []

    for filepath in glob.iglob(f'{path}/negatives/*.png'):
        img = mpimg.imread(filepath)
        for i in range(crops_per_negative):
            p = extract_patch(img)
            x.append(p)
            y.append(0)

    for filepath in glob.iglob(f'{path}/positives/*.png'):
        if 'mirr0r' in filepath and not include_mirrors:
            pass
        img = mpimg.imread(filepath)
        x.append(img)
        y.append(1)

    return np.asarray(x), np.asarray(y)


x, y = create_classification_dataset(
    'dataset/asl_eth_flir/flir_17_Sept_2013/train', 100)

print(x.shape, y.shape)
plt.imshow(x[0])
plt.show()
