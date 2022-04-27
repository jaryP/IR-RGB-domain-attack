import torch
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import draw_bounding_boxes


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_boxes(images, targets):
    imgs = [(im.detach().cpu() * 255).type(torch.ByteTensor) for im in images]

    boxes = [t['boxes'].detach().cpu() for t in targets]

    drawn_boxes = [draw_bounding_boxes(i, torch.round(b).type(torch.ByteTensor),
                                       colors="red") for i, b in zip(imgs, boxes)]
    show(drawn_boxes)
    plt.show()
