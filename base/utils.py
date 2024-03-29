import csv
import os

import numpy as np
import torch
import torchvision
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader, \
    ImageFolder
from torchvision.models import alexnet, vgg11, vgg16, resnet18, resnet34, \
    resnet50, resnext50_32x4d, resnext101_32x8d
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, transforms
from tqdm import tqdm
from pathlib import Path

from base.ir import load_nir_dataset, PathDataset
from models.alexnet import AlexNet
from models.resnet import resnet20


class TinyImagenet(Dataset):
    """Tiny Imagenet Pytorch Dataset,
    based on Avalanche implementation: https://github.com/ContinualAI/avalanche/
    blob/4763ceacd1ab961167d1a1deddbf88a9a10220a0/avalanche/benchmarks/datasets/
    tiny_imagenet/tiny_imagenet.py"""

    filename = ('tiny-imagenet-200.zip',
                'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(
            self,
            root,
            *,
            train: bool = True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=True):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        self.root = Path(root).expanduser()

        self.data_folder = self.root / 'tiny-imagenet-200'

        label2id = {}
        id2label = {}

        with open(str(os.path.join(self.data_folder, 'wnids.txt')), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        self.label2id, self.id2label = label2id, id2label

        self.data, self.targets = self.load_data()

    @staticmethod
    def labels2dict(data_folder):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.
        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(str(os.path.join(data_folder, 'wnids.txt')), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.
        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """
        Gets the training set image paths.
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """

        train_img_folder = self.data_folder / 'train' / class_name / 'images'

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name):
        """
        Gets the test set image paths
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        val_img_folder = self.data_folder / 'val' / 'images'
        annotations_file = self.data_folder / 'val' / 'val_annotations.txt'

        valid_names = []
        with open(str(annotations_file), 'r') as f:

            reader = csv.reader(f, dialect='excel-tab')
            for ll in reader:
                if ll[1] == class_name:
                    valid_names.append(ll[0])

        img_paths = [val_img_folder / f for f in valid_names]

        return img_paths

    def __len__(self):
        """ Returns the length of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_dataset(name, model_name, resize_dimensions=None,
                augmentation=True, path=None):
    if path is None:
        if name == 'imagenet':
            dataset_base_path = 'var/datasets/imagenet/'
        dataset_base_path = '~/datasets/'
    else:
        dataset_base_path = path

    dataset_base_path = os.path.expanduser(dataset_base_path)

    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root=dataset_base_path,
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root=dataset_base_path,
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   # Normalize(mn, std)
                   ])

        t = [
            ToTensor(),
            # Normalize(mn, std)
        ]

        if resize_dimensions is not None and isinstance(resize_dimensions, int):
            tt.append(Resize(resize_dimensions))
            t.append(Resize(resize_dimensions))

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.SVHN(
            root='~/datasets/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/datasets/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([
            ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        t = [
            ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]

        if resize_dimensions is not None and isinstance(resize_dimensions, int):
            tt.append(Resize(resize_dimensions))
            t.append(Resize(resize_dimensions))

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'tinyimagenet':
        tt = [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ]

        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ]

        if resize_dimensions is not None and isinstance(resize_dimensions, int):
            tt.append(Resize(resize_dimensions))
            t.append(Resize(resize_dimensions))

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        train_set = TinyImagenet('~/datasets/',
                                 transform=train_transform, train=True)

        test_set = TinyImagenet('~/datasets/',
                                transform=transform, train=False)

        input_size, classes = 3, 200

    elif name == 'imagenet':
        train_set = None

        test_set = ImageFolder(os.path.join(dataset_base_path, 'val'),
                               transform=Compose([transforms.ToTensor(),
                                                  Resize((256, 256))]))

        input_size = (3, 256, 256)
        classes = 1000

    elif 'nir' in name:
        input_size = (420, 420)
        classes = 9

        tt, t = [], []

        if resize_dimensions is not None and isinstance(resize_dimensions, int):
            # tt.append(Resize((resize_dimensions, resize_dimensions)))
            t.append(Resize((resize_dimensions, resize_dimensions)))
            input_size = (resize_dimensions, resize_dimensions)
        else:
            resize_dimensions = input_size
            # tt.append(Resize(resize_dimensions))
            t.append(Resize(resize_dimensions))

        tt.append(
            transforms.RandomCrop(resize_dimensions,
                                  padding=28))
        if augmentation:
                tt.append(RandomHorizontalFlip())

        tt.extend([ToTensor(),
                   transforms.Normalize(0.5, 0.5)])

        t.extend([ToTensor(),
                  transforms.Normalize(0.5, 0.5)])

        transform = Compose(t)
        train_transform = Compose(tt)

        _, ir, rgb = load_nir_dataset(dataset_base_path)

        np.random.seed(0)
        n = len(ir)
        n_test = int(0.1 * n)
        idxs = np.arange(n)
        np.random.shuffle(idxs)

        if name == 'ir_nir':
            train_set = PathDataset(paths=[ir[i] for i in idxs[n_test:]],
                                    is_ir=True, transform=train_transform)
            test_set = PathDataset(paths=[ir[i] for i in idxs[:n_test]],
                                   is_ir=True, transform=transform)
        elif name == 'rgb_nir':
            train_set = PathDataset(paths=[rgb[i] for i in idxs[n_test:]],
                                    is_ir=False, transform=train_transform)
            test_set = PathDataset(paths=[rgb[i] for i in idxs[:n_test]],
                                   is_ir=False, transform=transform)
        else:
            assert False
    else:
        assert False, f'{name} is not a valid NIR dataset. ' \
                      f'Accepted datasets are: ir_nir or rgb_nir.'

    return train_set, test_set, input_size, classes


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0):
    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam or sgd')


def model_training(model: nn.Module,
                   epochs: int,
                   optimizer: Optimizer,
                   dataloader: DataLoader):
    device = next(model.parameters()).device

    for epoch in tqdm(range(epochs)):
        model.train()

        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = nn.functional.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def get_model(name, image_size, classes, pre_trained=False,
              is_imagenet=False):
    name = name.lower()

    pre_trained = pre_trained or is_imagenet

    if name == 'alexnet':

        if is_imagenet:
            return alexnet(pretrained=True)
        else:
            return AlexNet(image_size[0], classes)

    if 'vgg' in name:
        if name == 'vgg11':
            model = vgg11(pretrained=pre_trained)
        elif name == 'vgg16':
            model = vgg16(pretrained=pre_trained)
        else:
            assert False

        if not is_imagenet:
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, classes)

    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20(classes)
        elif name == 'resnet18':
            model = resnet18(pretrained=pre_trained)
        elif name == 'resnet34':
            model = resnet34(pretrained=pre_trained)
        elif name == 'resnet50':
            model = resnet50(pretrained=pre_trained)
        else:
            assert False

        if not is_imagenet:
            model.fc = nn.Linear(
                model.fc.in_features, classes)

    elif 'resnext' in name:
        weights = None
        if pre_trained:
            weights = 'DEFAULT'
        if name == 'resnext50_32x4d':
            model = resnext50_32x4d(weights=weights)
        elif name == 'resnext101_32x8d':
            model = resnext101_32x8d(weights=weights)
        else:
            assert False

        if not is_imagenet:
            model.fc = nn.Linear(
                model.fc.in_features, classes)

    elif 'convnext' in name:
        if pre_trained:
            weights = 'DEFAULT'
        else:
            weights = None

        if name == 'convnext_tiny':
            model = torchvision.models.convnext_tiny(weights)
        else:
            assert False
        if not is_imagenet:
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, classes)
    else:
        assert False

    return model


