from itertools import chain

import numpy as np
import torch
from torch import nn, softmax
from torch import optim
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging

from torchattacks.attack import Attack

from attacks.base import get_attack
from base.adv import adv_train, adv_testing, adv_validation
from base.utils import get_model, get_dataset
from configs.nir_config import NIRConfig


class RandomWhitePixle(Attack):

    def __init__(self, model,
                 average_channels=True,
                 x_dimensions=(2, 10),
                 y_dimensions=(2, 10),
                 pixel_mapping='random',
                 restarts=20,
                 max_iterations=100,
                 update_each_iteration=False):

        super().__init__("Pixle", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        self.update_each_iteration = update_each_iteration
        self.max_patches = max_iterations

        self.restarts = restarts
        self.pixel_mapping = pixel_mapping.lower()
        self.average_channels = average_channels

        if self.pixel_mapping not in ['random', 'similarity',
                                      'similarity_random', 'distance',
                                      'distance_random']:
            raise ValueError('pixel_mapping must be one of [random, similarity,'
                             'similarity_random, distance, distance_random]'
                             ' ({})'.format(self.pixel_mapping))

        if isinstance(y_dimensions, (int, float)):
            y_dimensions = [y_dimensions, y_dimensions]

        if isinstance(x_dimensions, (int, float)):
            x_dimensions = [x_dimensions, x_dimensions]

        if not all([(isinstance(d, (int)) and d > 0)
                    or (isinstance(d, float) and 0 <= d <= 1)
                    for d in chain(y_dimensions, x_dimensions)]):
            raise ValueError('dimensions of first patch must contains integers'
                             ' or floats in [0, 1]'
                             ' ({})'.format(y_dimensions))

        self.p1_x_dimensions = x_dimensions
        self.p1_y_dimensions = y_dimensions

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, return_solutions=False):
        n_im, c, h, w = images.shape

        x_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(w * d))
             for d in self.p1_x_dimensions])

        y_bounds = tuple(
            [max(1, d if isinstance(d, int) else round(h * d))
             for d in self.p1_y_dimensions])

        images = images.to(self.device)
        labels = labels.to(self.device)

        images.requires_grad = True

        loss = cross_entropy(self.model(images), labels)
        self.model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        data_grad = torch.abs(data_grad)

        if self.average_channels:
            data_grad = data_grad.mean(1)
            shape = (h, w)
        else:
            shape = (c, h, w)

        data_grad = data_grad.view(n_im, -1)
        indexes = torch.argsort(data_grad, -1).view(n_im, -1)
        # probs = torch.exp(data_grad) / \
        #         torch.exp(data_grad).sum(-1, keepdim=True)

        probs = data_grad / data_grad.sum(-1, keepdim=True)

        probs = probs.detach().cpu().numpy()
        indexes = indexes.detach().cpu().numpy()

        adv_images = []
        swapped_pixels = []

        for img_i in range(len(images)):
            img = images[img_i]
            img_indexes = indexes[img_i]

            img_probs = probs[img_i]
            label = labels[img_i]

            loss, callback = self._get_fun(img, label,
                                           target_attack=False)

            best_adv_image = img.clone()
            image_swapped_pixels = []

            for restart_i in range(self.restarts):
                stop = False
                best_solution = None
                best_loss = loss(best_adv_image)

                for patch_i in range(self.max_patches):
                    pert_image = best_adv_image.clone()

                    (x, y), (x_offset, y_offset) = \
                        self.get_patch_coordinates(image=img,
                                                   x_bounds=x_bounds,
                                                   y_bounds=y_bounds)

                    pixels_to_take = x_offset * y_offset
                    selected_indexes = np.random.choice(img_indexes,
                                                        pixels_to_take,
                                                        True, img_probs)

                    pixels_places = [np.unravel_index(i, shape)
                                     for i in selected_indexes]

                    it_pixels_places = iter(pixels_places)

                    for _ix in range(x_offset):
                        for _iy in range(y_offset):
                            b = (x + _ix, y + _iy)
                            a = next(it_pixels_places)

                            if self.average_channels:
                                # adv_img[:, a] = img[:, b]
                                # v = adv_img[:, b[0], b[1]]
                                pert_image[:, b[0], b[1]] = img[:, a[0], a[1]]
                                # if self.swap:
                                #     adv_img[:, a[0], a[1]] = v
                            else:
                                # v = adv_img[b[0], b[1], b[2]]
                                pert_image[b[0], b[1], b[2]] = img[
                                    a[0], a[1], a[2]]
                                # if self.swap:
                                #     adv_img[a[0], a[1], a[2]] = v

                    l = loss(pert_image)

                    if l < best_loss:
                        best_loss = l
                        best_solution = (
                            (x, y), (x_offset, y_offset), pixels_places)

                    if callback(pert_image, None, True):
                        best_solution = (
                            (x, y), (x_offset, y_offset), pixels_places)
                        stop = True
                        break

                if best_solution is not None:
                    image_swapped_pixels.append(best_solution)

                    (x, y), (x_offset, y_offset), pixels_places = best_solution

                    it_pixels_places = iter(pixels_places)

                    for _ix in range(x_offset):
                        for _iy in range(y_offset):
                            b = (x + _ix, y + _iy)
                            a = next(it_pixels_places)

                            if self.average_channels:
                                # adv_img[:, a] = img[:, b]
                                # v = adv_img[:, b[0], b[1]]
                                best_adv_image[:, b[0], b[1]] = img[:, a[0],
                                                                a[1]]
                                # if self.swap:
                                #     adv_img[:, a[0], a[1]] = v
                            else:
                                # v = adv_img[b[0], b[1], b[2]]
                                best_adv_image[b[0], b[1], b[2]] = img[
                                    a[0], a[1], a[2]]
                                # if self.swap:
                                #     adv_img[a[0], a[1], a[2]] = v

                if stop:
                    break

            swapped_pixels.append(image_swapped_pixels)
            adv_images.append(best_adv_image)

        adv_images = torch.stack(adv_images, 0)

        if return_solutions:
            return adv_images, swapped_pixels

        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def loss(self, img, label, target_attack=False):

        p = self._get_prob(img)
        p = p[np.arange(len(p)), label]

        if target_attack:
            p = 1 - p

        return p.sum()

    def get_patch_coordinates(self, image, x_bounds, y_bounds):
        c, h, w = image.shape

        x, y = np.random.uniform(0, 1, 2)

        x_offset = np.random.randint(x_bounds[0],
                                     x_bounds[1] + 1)

        y_offset = np.random.randint(y_bounds[0],
                                     y_bounds[1] + 1)

        x, y = int(x * (w - 1)), int(y * (h - 1))

        if x + x_offset > w:
            x_offset = w - x

        if y + y_offset > h:
            y_offset = h - y

        return (x, y), (x_offset, y_offset)

    def get_pixel_mapping(self, source_image, x, x_offset, y, y_offset,
                          destination_image=None):
        if destination_image is None:
            destination_image = source_image

        destinations = []
        c, h, w = source_image.shape[1:]
        source_image = source_image[0]

        if self.pixel_mapping == 'random':
            for i in range(x_offset):
                for j in range(y_offset):
                    dx, dy = np.random.uniform(0, 1, 2)
                    dx, dy = int(dx * (w - 1)), int(dy * (h - 1))
                    destinations.append([dx, dy])
        else:
            for i in np.arange(y, y + y_offset):
                for j in np.arange(x, x + x_offset):
                    pixel = source_image[:, i: i + 1, j: j + 1]
                    diff = destination_image - pixel
                    diff = diff[0].abs().mean(0).view(-1)

                    if 'similarity' in self.pixel_mapping:
                        diff = 1 / (1 + diff)
                        diff[diff == 1] = 0

                    probs = torch.softmax(diff, 0).cpu().numpy()

                    indexes = np.arange(len(diff))

                    pair = None

                    linear_iter = iter(sorted(zip(indexes, probs),
                                              key=lambda pit: pit[1],
                                              reverse=True))

                    while True:
                        if 'random' in self.pixel_mapping:
                            index = np.random.choice(indexes, p=probs)
                        else:
                            index = next(linear_iter)[0]

                        _y, _x = np.unravel_index(index, (h, w))

                        if _y == i and _x == j:
                            continue

                        pair = (_x, _y)
                        break

                    destinations.append(pair)

        return destinations

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution,
                 destination=None,
                 solution_as_perturbed=False, **kwargs):

            # if not solution_as_perturbed:
            #     pert_image = self._perturb(source=img,
            #                                destination=destination,
            #                                solution=solution)
            # else:
            #     pert_image = solution

            if len(solution.shape) == 3:
                solution = solution[None, :]

            p = self._get_prob(solution)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(solution,
                     destination=None,
                     solution_as_perturbed=False,
                     **kwargs):

            # if not solution_as_perturbed:
            #     pert_image = self._perturb(source=img,
            #                                destination=destination,
            #                                solution=solution)
            # else:
            #     pert_image = solution
            if len(solution.shape) == 3:
                solution = solution[None, :]

            p = self._get_prob(solution)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
            else:
                return mx != label

        return func, callback

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source

        c, h, w = source.shape[1:]

        x, y, xl, yl = solution[:4]
        destinations = solution[4:]

        source_pixels = np.ix_(range(c),
                               np.arange(y, y + yl),
                               np.arange(x, x + xl))

        indexes = torch.tensor(destinations)
        destination = destination.clone().detach().to(self.device)

        s = source[0][source_pixels].view(c, -1)

        destination[0, :, indexes[:, 0], indexes[:, 1]] = s

        return destination


logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="NIR_config", node=NIRConfig)


@hydra.main(config_path="configs", config_name="base_adv_config")
def main(cfg: NIRConfig):
    print(OmegaConf.to_yaml(cfg))

    device = 'cpu'
    if cfg.train_params.device != 'cpu' and torch.cuda.is_available():
        device = f'cuda:{cfg.train_params.device}'

    train_set, test_set, input_size, classes = get_dataset(
        name=cfg.dataset.name, model_name=None,
        path=cfg.dataset.path,
        resize_dimensions=cfg.params.img_size)

    train, val, test = cfg.params.train, cfg.params.val, cfg.params.test

    # if isinstance(train, float):
    #     train = int(train * len(dataset))

    if isinstance(val, float):
        val = int(val * len(train_set))

    # if isinstance(test, float):
    #     test = int(test * len(dataset))

    train = len(train_set) - val

    train_set, val_set = torch.utils.data.random_split(train_set, [
        train, val])

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.params.batch_size,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=cfg.params.batch_size,
                                              shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=cfg.params.batch_size,
                                             shuffle=False, num_workers=0)

    model = get_model(cfg.model.name, input_size, classes,
                      True)
    model.to(device)

    # attack = get_attack(cfg.attack_type.name, model, x=cfg.param_attack.x_dim,
    #                     y=cfg.param_attack.y_dim,
    #                     restarts=cfg.param_attack.rest,
    #                     max_iterations=cfg.param_attack.max_iter)

    attack = RandomWhitePixle(model=model,
                              x_dimensions=cfg.params_attack.x_dim,
                              y_dimensions=cfg.params_attack.y_dim,
                              restarts=cfg.params_attack.rest,
                              max_iterations=cfg.params_attack.max_iter)

    print(cfg.model.name)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_params.lr)

    for epoch in range(cfg.train_params.epoch_count):
        adv_train(epoch, train_loader, model, optimizer, loss,
                  cfg.train_params.log_freq,
                  attack=attack,
                  type_of_attack='white',
                  attack_per_batch=cfg.params_attack.attack_per_batch)

        adv_validation(model, val_loader, loss, attack=attack,
                       type_of_attack='white',
                       attack_per_batch=cfg.params_attack.attack_per_batch)

    adv_testing(model, test_loader, attack=attack,
                type_of_attack='white',
                attack_per_batch=cfg.params_attack.attack_per_batch)


if __name__ == '__main__':
    main()
