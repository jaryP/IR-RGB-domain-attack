from itertools import chain

import numpy as np
import torch
from torch.nn.functional import softmax, cross_entropy
from torchattacks.attack import Attack


class Node:
    def __init__(self, parent_node, source_pixel, destination_pixel):
        self.current_loss = np.inf

        if parent_node is not None:
            self.parent_node = parent_node
            self.depth = parent_node.depth + 1

            self.source_pixel = source_pixel
            self.destination_pixel = destination_pixel

        self.child = []

        self.prob = None
        self.prediction = None

    def add_child(self, c):
        self.child.append(c)

    def calculate_prediction_and_loss(self, image, model):
        adv_img = image.copy()

        adv_img[:, self.source_pixel[0], self.source_pixel[1]] = \
            image[:, self.destination_pixel[0], self.destination_pixel[1]]

        output = softmax(model(adv_img[None, :]), -1)
        pred = output.argmax(-1)
        mx = output.max(-1)

        self.prob = mx
        self.prediction = pred

        return pred[0], mx[0]


def iterative_tree_search(image, label, model, max_depth, max_iterations):
    c, h, w = image.shape

    root = Node(None, None, None)

    image.requires_grad = True

    loss = cross_entropy(model(image), label)
    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data

    data_grad = data_grad.mean(1)
    shape = (h, w)

    data_grad = torch.abs(data_grad)

    data_grad = torch.flatten(data_grad, 1)
    indexes = torch.argsort(data_grad, -1)


def expand_node(current_node, image, label, model, indexes, selected_indexes):
    p = np.ones_like(indexes)
    p[selected_indexes] = 0

    source_pixel, destination_pixel = np.random.choice(indexes, 2,
                                                       False, p / p.sum())

    source_pixel = np.unravel_index(source_pixel, (32, 32))
    destination_pixel = np.unravel_index(destination_pixel, (32, 32))

    node = Node(parent_node=current_node,
                source_pixel=source_pixel,
                destination_pixel=destination_pixel)

    node.calculate_prediction_and_loss(image, model)

    if node.




def populate_tree(image, label, model, max_depth, max_iterations):
    c, h, w = image.shape

    root = Node(None, None, None)

    image.requires_grad = True

    loss = cross_entropy(model(image), label)
    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data

    data_grad = data_grad.mean(1)
    shape = (h, w)

    data_grad = torch.abs(data_grad)

    data_grad = torch.flatten(data_grad, 1)
    indexes = torch.argsort(data_grad, -1)

    frontier = [root]
    frontier_images = [image.copy()]
    frontier_masks = [np.ones_like(indexes)]

    while frontier:
        current_node = frontier.pop()





class WhitePixle(Attack):
    def __init__(self, model,
                 swap=False,
                 average_channels=True,
                 attack_limit=-1,
                 descending: bool = False):

        super().__init__("WhitePixle", model)

        self.swap = swap
        self.average_channels = average_channels
        self.attack_limit = attack_limit
        self.descending = descending

        # self._supported_mode = ['default', 'targeted']
        self._supported_mode = ['default']

    def forward(self, images, labels):
        n_im, c, h, w = images.shape

        images = images.to(self.device)
        labels = labels.to(self.device)

        images.requires_grad = True

        loss = cross_entropy(self.model(images), labels)
        self.model.zero_grad()
        loss.backward()

        data_grad = images.grad.data

        if self.average_channels:
            data_grad = data_grad.mean(1)
            shape = (h, w)
        else:
            shape = (c, h, w)

        data_grad = torch.abs(data_grad)

        data_grad = torch.flatten(data_grad, 1)
        indexes = torch.argsort(data_grad, -1, descending=self.descending)

        adv_images = []
        for img_i in range(len(images)):
            img = images[img_i]
            adv_img = img.clone()
            img_target = labels[img_i]

            img_grads = data_grad[img_i]
            img_indexes = indexes[img_i]

            for i in range(len(img_indexes) // 2):
                if 0 < self.attack_limit < i:
                    break

                less_important, more_important = img_indexes[i], img_indexes[
                    - (i + 1)]

                a = np.unravel_index(less_important.item(), shape)
                b = np.unravel_index(more_important.item(), shape)

                if self.average_channels:
                    # adv_img[:, a] = img[:, b]
                    v = adv_img[:, b[0], b[1]]
                    adv_img[:, b[0], b[1]] = img[:, a[0], a[1]]
                    if self.swap:
                        adv_img[:, a[0], a[1]] = v
                else:
                    v = adv_img[b[0], b[1], b[2]]
                    adv_img[b[0], b[1], b[2]] = img[a[0], a[1], a[2]]
                    if self.swap:
                        adv_img[a[0], a[1], a[2]] = v

                output = self.model(adv_img[None, :])
                pred = output.argmax(-1)

                if img_target.item() != pred.item():
                    # modified_pixels.append(i + 1)
                    break

            adv_images.append(adv_img)

        adv_images = torch.stack(adv_images, 0)
        return adv_images
