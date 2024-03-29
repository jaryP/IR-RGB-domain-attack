from itertools import chain
from typing import Sequence

import numpy as np
import torch
from torch.nn.functional import softmax, cross_entropy
from torchattacks.attack import Attack


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

    def forward(self, images, labels, return_swaps=False):
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
        all_swaps = []

        for img_i in range(len(images)):
            img = images[img_i]
            adv_img = img.clone()
            img_target = labels[img_i]

            img_grads = data_grad[img_i]
            img_indexes = indexes[img_i]

            image_swaps = []

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


class PatchWhitePixle(Attack):

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
        self.max_iterations = max_iterations

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

        indexes = torch.argsort(data_grad, -1).view(n_im, -1)
        data_grad = data_grad.view(n_im, -1)
        # probs = torch.exp(data_grad) / \
        #         torch.exp(data_grad).sum(-1, keepdim=True)

        probs = data_grad / data_grad.sum(-1, keepdim=True)

        probs = probs.detach().cpu().numpy()
        indexes = indexes.detach().cpu().numpy()

        adv_images = []
        swapped_pixels = []
        iterations = []
        statistics = []
        image_probs = []

        for img_i in range(len(images)):
            img = images[img_i]
            img_indexes = indexes[img_i]

            img_probs = probs[img_i]
            label = labels[img_i]

            loss, callback = self._get_fun(img, label,
                                           target_attack=False)

            best_adv_image = img.clone()
            image_swapped_pixels = []

            im_iterations = 0

            for restart_i in range(self.restarts):

                stop = False
                best_solution = None
                best_loss = loss(best_adv_image)

                patch_i = 0
                for patch_i in range(self.max_iterations):
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

                    image_probs.append(best_loss)

                    if callback(pert_image, None, True):
                        best_solution = (
                            (x, y), (x_offset, y_offset), pixels_places)
                        stop = True
                        break

                im_iterations += patch_i
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

            iterations.append(im_iterations)
            statistics.append(image_probs)

            swapped_pixels.append(image_swapped_pixels)
            adv_images.append(best_adv_image.detach())

        self.probs_statistics = statistics
        self.required_iterations = iterations

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


class RandomWhitePixle(Attack):

    def __init__(self, model,
                 pixels_per_iteration=1,
                 mode='htl',
                 average_channels=True,
                 acceptance_threshold=1,
                 restarts=20,
                 max_iterations=100,
                 **kwargs):

        super().__init__("Pixle", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        assert mode in ['htl', 'lth']

        self.mode = mode
        self.iterations = max_iterations
        self.pixels_per_iteration = pixels_per_iteration

        self.restarts = restarts
        self.average_channels = average_channels

        self.acceptance_threshold = acceptance_threshold
        assert 0 < acceptance_threshold <= 1, 'acceptance_threshold must be in (0, 1]'

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, return_solutions=False):

        shape = images.shape

        if len(shape) == 3:
            images = images[None]
            c, h, w = shape
        else:
            _, c, h, w = shape

        images = images.to(self.device)
        labels = labels.to(self.device)

        images.requires_grad = True

        adv_images = []
        swapped_pixels = []
        iterations = []
        statistics = []
        image_probs = []

        for img_i in range(len(images)):
            img = images[img_i]
            label = labels[img_i]

            loss_f, callback_f = self._get_fun(label, target_attack=False)

            best_adv_image = img.clone()
            image_swapped_pixels = []

            im_iterations = 0

            initial_loss = loss_f(img)

            for restart_i in range(self.restarts):

                loss = cross_entropy(self.model(img[None]), label[None],
                                     reduction='none')
                self.model.zero_grad()
                img.grad = None

                data_grad = torch.autograd.grad(loss, img,
                                                retain_graph=False,
                                                create_graph=False)[0]

                data_grad = torch.abs(data_grad)

                if self.average_channels:
                    data_grad = data_grad.mean(0)
                    shape = (h, w)
                else:
                    shape = (c, h, w)

                if isinstance(self.pixels_per_iteration, float):
                    pixels_per_iteration = int(
                        self.pixels_per_iteration * (h * w))
                elif isinstance(self.pixels_per_iteration, Sequence):
                    a, b = self.pixels_per_iteration

                    if isinstance(a, float):
                        a = int(self.pixels_per_iteration[0] * (h * w))

                    if isinstance(b, float):
                        b = int(self.pixels_per_iteration[1] * (h * w))

                    if b < a:
                        c = a
                        a = b
                        b = a

                    pixels_per_iteration = np.random.randint(a, b + 1)
                else:
                    pixels_per_iteration = self.pixels_per_iteration

                data_grad = data_grad.view(-1)
                probs = data_grad / data_grad.sum(-1, keepdim=True)

                data_grad = 1 / data_grad
                data_grad = torch.nan_to_num(data_grad, posinf=0.0, neginf=0.0)

                invert_probs = data_grad / data_grad.sum(-1, keepdim=True)

                if self.mode == 'htl':
                    source_prob = probs
                    dest_prob = invert_probs
                else:
                    source_prob = invert_probs
                    dest_prob = probs

                indexes = np.arange(len(source_prob))

                source_prob = source_prob.detach().cpu().numpy()
                dest_prob = dest_prob.detach().cpu().numpy()

                # source_prob = np.nan_to_num(source_prob, posinf=0.0, neginf=0.0)
                # dest_prob = np.nan_to_num(dest_prob, posinf=0.0, neginf=0.0)

                # if dest_prob.sum() == 0.0 or source_prob.sum() == 0.0:
                #     break

                stop = False
                best_solution = None
                best_loss = loss_f(best_adv_image)

                patch_i = 0

                for patch_i in range(self.iterations):
                    pert_image = best_adv_image.clone()

                    selected_indexes1 = np.random.choice(indexes,
                                                         pixels_per_iteration,
                                                         False, source_prob)

                    selected_indexes2 = np.random.choice(indexes,
                                                         pixels_per_iteration,
                                                         False, dest_prob)

                    aa = torch.tensor([np.unravel_index(_a, shape) for _a in
                                       selected_indexes1], device=self.device)

                    bb = torch.tensor([np.unravel_index(_b, shape) for _b in
                                       selected_indexes2], device=self.device)

                    # best_adv_image[:, aa[:, 0], aa[:, 1]] = \
                    #     img[:, bb[:, 0], bb[:, 1]]

                    # for a, b in zip(aa, bb):
                    if self.average_channels:
                        # best_adv_image[:, bb[0], bb[1]] = img[:, aa[0], aa[1]]
                        pert_image[:, aa[:, 0], aa[:, 1]] = \
                            img[:, bb[:, 0], bb[:, 1]]
                    else:
                        # best_adv_image[bb[0], b[1], b[2]] = img[
                        #     a[0], a[1], a[2]]
                        # if self.swap:
                        #     adv_img[a[0], a[1], a[2]] = v
                        pert_image[aa[:, 0], aa[:, 1], aa[:, 2]] = \
                            img[bb[:, 0], bb[:, 1], bb[:, 2]]

                    l = loss_f(pert_image)

                    if l < best_loss:
                        best_loss = l
                        best_solution = (aa, bb)
                        best_adv_image = pert_image.clone()

                    image_probs.append(best_loss)

                    if callback_f(pert_image) \
                            and best_loss < self.acceptance_threshold * initial_loss:
                        best_solution = (aa, bb)
                        best_adv_image = pert_image.clone()
                        stop = True
                        break

                im_iterations += patch_i
                image_swapped_pixels.append(best_solution)

                if stop or best_solution is None:
                    break
                else:
                    img = best_adv_image.clone()

            iterations.append(im_iterations)
            statistics.append(image_probs)

            swapped_pixels.append(image_swapped_pixels)
            adv_images.append(best_adv_image.detach())

        self.probs_statistics = statistics
        self.required_iterations = iterations

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

    def _get_fun(self, label, target_attack=False):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)[0]
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


class BinaryRandomWhitePixle(Attack):
    def __init__(self, model,
                 pixels_per_iteration=1,
                 mode='htl',
                 average_channels=True,
                 restarts=20,
                 iterations=100,
                 **kwargs):

        super().__init__("Pixle", model)

        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError('restarts must be and integer >= 0 '
                             '({})'.format(restarts))

        assert mode in ['htl', 'lth']

        self.mode = mode
        self.iterations = iterations
        self.pixels_per_iteration = pixels_per_iteration

        self.restarts = restarts
        self.average_channels = average_channels

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, return_solutions=False):

        shape = images.shape

        if len(shape) == 3:
            images = images[None]
            c, h, w = shape
        else:
            _, c, h, w = shape

        images = images.to(self.device)
        labels = labels.to(self.device)

        images.requires_grad = True

        adv_images = []
        swapped_pixels = []
        iterations = []
        statistics = []
        image_probs = []

        for img_i in range(len(images)):
            img = images[img_i]
            label = labels[img_i]

            loss_f, callback_f = self._get_fun(label, target_attack=False)

            best_adv_image = img.clone()
            image_swapped_pixels = []

            im_iterations = 0

            loss = cross_entropy(self.model(img[None]), label[None],
                                 reduction='none')
            self.model.zero_grad()
            img.grad = None

            data_grad = torch.autograd.grad(loss, img,
                                            retain_graph=False,
                                            create_graph=False)[0]

            data_grad = torch.abs(data_grad)

            if self.average_channels:
                data_grad = data_grad.mean(0)
                shape = (h, w)
            else:
                shape = (c, h, w)

            if isinstance(self.pixels_per_iteration, float):
                pixels_per_iteration = int(
                    self.pixels_per_iteration * (h * w))
            else:
                pixels_per_iteration = self.pixels_per_iteration

            data_grad = data_grad.view(-1)
            probs = data_grad / data_grad.sum(-1, keepdim=True)

            data_grad = 1 / data_grad
            data_grad = torch.nan_to_num(data_grad, posinf=0.0, neginf=0.0)

            invert_probs = data_grad / data_grad.sum(-1, keepdim=True)

            if self.mode == 'htl':
                source_prob = probs
                dest_prob = invert_probs
            else:
                source_prob = invert_probs
                dest_prob = probs

            indexes = np.arange(len(source_prob))

            source_prob = source_prob.detach().cpu().numpy()
            dest_prob = dest_prob.detach().cpu().numpy()

            split = 0.5
            best_solution = None
            best_loss = loss_f(best_adv_image)
            adv_image = None

            tot_it = 0
            it = 0

            while True:
                best_loss_it = best_loss
                pixels_per_iteration = int(split * (h * w))
                found = False

                if pixels_per_iteration >= (h * w):
                    break

                for it in range(self.iterations):
                    pert_image = img.clone()

                    selected_indexes1 = np.random.choice(indexes,
                                                         pixels_per_iteration,
                                                         False, source_prob)

                    selected_indexes2 = np.random.choice(indexes,
                                                         pixels_per_iteration,
                                                         False, dest_prob)

                    aa = torch.tensor([np.unravel_index(_a, shape) for _a in
                                       selected_indexes1], device=self.device)

                    bb = torch.tensor([np.unravel_index(_b, shape) for _b in
                                       selected_indexes2], device=self.device)

                    pert_image[:, bb[:, 0], bb[:, 1]] = \
                        img[:, aa[:, 0], aa[:, 1]]

                    # for a, b in zip(aa, bb):
                    #     if self.average_channels:
                    #         pert_image[:, b[0], b[1]] = img[:, a[0], a[1]]
                    #     else:
                    #         pert_image[b[0], b[1], b[2]] = img[
                    #             a[0], a[1], a[2]]

                    l = loss_f(pert_image)

                    # if l < best_loss_it:
                    if callback_f(pert_image):
                        best_loss_it = l
                        best_solution = (aa, bb)
                        found = True

                        best_adv_image = img.clone()
                        best_adv_image[:, aa] = best_adv_image[:, bb]

                        break

                    if callback_f(pert_image):
                        break

                statistics.append(best_loss_it)

                # if best_loss_it < best_loss:
                if found:
                    # aa, bb = best_solution
                    # best_adv_image = img.clone()
                    #
                    # best_adv_image[:, aa] = best_adv_image[:, bb]
                    #
                    # if callback_f(best_adv_image):
                    #     adv_image = best_adv_image.clone()
                    # else:
                    #     break

                    if pixels_per_iteration == 1:
                        break

                    split = split - (split / 2)
                else:
                    if adv_image is not None:
                        break

                    split = split + (split / 2)

                tot_it += it + 1

            iterations.append(tot_it)
            adv_images.append(adv_image if adv_image is not None else img)

        #     for restart_i in range(self.restarts):
        #
        #         loss = cross_entropy(self.model(img[None]), label[None],
        #                              reduction='none')
        #         self.model.zero_grad()
        #         img.grad = None
        #
        #         data_grad = torch.autograd.grad(loss, img,
        #                                         retain_graph=False,
        #                                         create_graph=False)[0]
        #
        #         data_grad = torch.abs(data_grad)
        #
        #         if self.average_channels:
        #             data_grad = data_grad.mean(0)
        #             shape = (h, w)
        #         else:
        #             shape = (c, h, w)
        #
        #         if isinstance(self.pixels_per_iteration, float):
        #             pixels_per_iteration = int(
        #                 self.pixels_per_iteration * (h * w))
        #         else:
        #             pixels_per_iteration = self.pixels_per_iteration
        #
        #         data_grad = data_grad.view(-1)
        #         probs = data_grad / data_grad.sum(-1, keepdim=True)
        #
        #         data_grad = 1 / data_grad
        #         data_grad = torch.nan_to_num(data_grad, posinf=0.0, neginf=0.0)
        #
        #         invert_probs = data_grad / data_grad.sum(-1, keepdim=True)
        #
        #         if self.mode == 'htl':
        #             source_prob = probs
        #             dest_prob = invert_probs
        #         else:
        #             source_prob = invert_probs
        #             dest_prob = probs
        #
        #         indexes = np.arange(len(source_prob))
        #
        #         source_prob = source_prob.detach().cpu().numpy()
        #         dest_prob = dest_prob.detach().cpu().numpy()
        #
        #         # source_prob = np.nan_to_num(source_prob, posinf=0.0, neginf=0.0)
        #         # dest_prob = np.nan_to_num(dest_prob, posinf=0.0, neginf=0.0)
        #
        #         # if dest_prob.sum() == 0.0 or source_prob.sum() == 0.0:
        #         #     break
        #
        #         stop = False
        #         best_solution = None
        #         best_loss = loss_f(best_adv_image)
        #
        #         patch_i = 0
        #
        #         for patch_i in range(self.iterations):
        #             pert_image = best_adv_image.clone()
        #
        #             selected_indexes1 = np.random.choice(indexes,
        #                                                  pixels_per_iteration,
        #                                                  False, source_prob)
        #
        #             selected_indexes2 = np.random.choice(indexes,
        #                                                  pixels_per_iteration,
        #                                                  False, dest_prob)
        #
        #             aa = [np.unravel_index(_a, shape) for _a in
        #                   selected_indexes1]
        #             bb = [np.unravel_index(_b, shape) for _b in
        #                   selected_indexes2]
        #
        #             for a, b in zip(aa, bb):
        #                 if self.average_channels:
        #                     pert_image[:, b[0], b[1]] = img[:, a[0], a[1]]
        #                     # if self.swap:
        #                     #     adv_img[:, a[0], a[1]] = v
        #                 else:
        #                     pert_image[b[0], b[1], b[2]] = img[
        #                         a[0], a[1], a[2]]
        #                     # if self.swap:
        #                     #     adv_img[a[0], a[1], a[2]] = v
        #
        #             l = loss_f(pert_image)
        #
        #             if l < best_loss:
        #                 best_loss = l
        #                 best_solution = (selected_indexes1, selected_indexes2)
        #
        #             image_probs.append(best_loss)
        #
        #             if callback_f(pert_image):
        #                 best_solution = (selected_indexes1, selected_indexes2)
        #                 stop = True
        #                 break
        #
        #         im_iterations += patch_i
        #         if best_solution is not None:
        #             image_swapped_pixels.append(best_solution)
        #
        #             selected_indexes1, selected_indexes2 = best_solution
        #
        #             aa = [np.unravel_index(_a, shape) for _a in
        #                   selected_indexes1]
        #             bb = [np.unravel_index(_b, shape) for _b in
        #                   selected_indexes2]
        #
        #             for a, b in zip(aa, bb):
        #                 if self.average_channels:
        #                     best_adv_image[:, b[0], b[1]] = img[:, a[0], a[1]]
        #                     # if self.swap:
        #                     #     adv_img[:, a[0], a[1]] = v
        #                 else:
        #                     best_adv_image[b[0], b[1], b[2]] = img[
        #                         a[0], a[1], a[2]]
        #                     # if self.swap:
        #                     #     adv_img[a[0], a[1], a[2]] = v
        #
        #             img = best_adv_image.clone()
        #
        #         if stop:
        #             break
        #
        #     iterations.append(im_iterations)
        #     statistics.append(image_probs)
        #
        #     swapped_pixels.append(image_swapped_pixels)
        #     adv_images.append(best_adv_image.detach())
        #

        self.probs_statistics = statistics
        self.required_iterations = iterations

        adv_images = torch.stack(adv_images, 0).detach()

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

    def _get_fun(self, label, target_attack=False):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p.sum()

        @torch.no_grad()
        def callback(img, **kwargs):

            if len(img.shape) == 3:
                img = img[None, :]

            p = self._get_prob(img)[0]
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
