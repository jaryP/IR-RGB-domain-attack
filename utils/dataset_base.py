import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseObjectDetectionDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_path, download=False, transforms=None,
                 target_transforms=None):

        self.root_path = os.path.abspath(root_path)
        self.transforms = transforms
        self.target_transforms = target_transforms

        if not self.check_existence():
            if not download:
                raise ValueError
            else:
                self.download()

        self.load_dataset()

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def download(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_existence(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_boxes(self, idx) -> np.ndarray:
        raise NotImplementedError

    def get_masks(self, idx) -> Union[np.ndarray, None]:
        return None

    @abstractmethod
    def get_image(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_labels(self, idx, image, masks, boxes):
        raise NotImplementedError

    @abstractmethod
    def is_crowd(self, idx, image, masks) -> bool:
        raise NotImplementedError

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        img = self.get_image(idx)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        masks = self.get_masks(idx)
        # instances are encoded as different colors
        # obj_ids = np.unique(masks)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        boxes = self.get_boxes(idx)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = self.get_labels(idx, img, masks, boxes)
        labels = torch.tensor(labels, dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        if len(boxes) == 0:
            area = torch.tensor([0])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        if not self.is_crowd(idx, img, labels):
            iscrowd = torch.zeros_like(labels, dtype=torch.int64)
        else:
            iscrowd = torch.ones_like(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            target["masks"] = masks

        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)

        return img, target

    #
    #
    # def __getitem__(self, idx):
    #     # load images and masks
    #     # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
    #     # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
    #     # img = Image.open(img_path).convert("RGB")
    #     img = self.get_image(idx)
    #     # note that we haven't converted the mask to RGB,
    #     # because each color corresponds to a different instance
    #     # with 0 being background
    #     # mask = Image.open(mask_path)
    #     # convert the PIL Image into a numpy array
    #     masks = self.get_masks(idx)
    #     # instances are encoded as different colors
    #     # obj_ids = np.unique(masks)
    #     # first id is the background, so remove it
    #     # obj_ids = obj_ids[1:]
    #
    #     # split the color-encoded mask into a set
    #     # of binary masks
    #     # masks = mask == obj_ids[:, None, None]
    #
    #     # get bounding box coordinates for each mask
    #     # num_objs = len(obj_ids)
    #     boxes = self.get_boxes(idx)
    #
    #     # convert everything into a torch.Tensor
    #     boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #     # there is only one class
    #     labels = self.get_labels(idx, img, masks, boxes)
    #     labels = torch.tensor(labels, dtype=torch.int64)
    #     # labels = torch.ones((num_objs,), dtype=torch.int64)
    #
    #     if len(boxes) == 0:
    #         area = torch.tensor([0])
    #     else:
    #         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    #
    #     # suppose all instances are not crowd
    #     if not self.is_crowd(idx, img, labels):
    #         iscrowd = torch.zeros_like(labels, dtype=torch.int64)
    #     else:
    #         iscrowd = torch.ones_like(labels, dtype=torch.int64)
    #
    #     target = {}
    #     target["boxes"] = boxes
    #     target["labels"] = labels
    #     if masks is not None:
    #         masks = torch.as_tensor(masks, dtype=torch.uint8)
    #         target["masks"] = masks
    #
    #     target["image_id"] = torch.tensor([idx])
    #     target["area"] = area
    #     target["iscrowd"] = iscrowd
    #
    #     if self.transforms is not None:
    #         img = self.transforms(img)
    #
    #     # if self.target_transforms is not None:
    #     #     target = self.target_transforms(target)
    #
    #     return img, target

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
