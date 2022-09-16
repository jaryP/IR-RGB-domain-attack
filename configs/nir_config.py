from dataclasses import dataclass
from typing import Union


@dataclass
class Paths:
    log: str


@dataclass
class Files:
    rgb: str


@dataclass
class Params:
    batch_size: int
    train: Union[float, int]
    val: Union[float, int]
    test: Union[float, int]
    classes: int
    img_size: int
    images_to_attack_per_label: int


@dataclass
class TrainParams:
    pre_training_epochs: int
    lr: float
    log_freq: int
    device: Union[str, int]
    adv_training_epochs: int
    pre_trained: bool
    eval_every_n_epochs: int


@dataclass
class Models:
    name: str


@dataclass
class Data:
    path: str
    name: str


@dataclass
class PatchWhiteBox:
    x_dim: int
    y_dim: int
    rest: int
    max_iter: int
    attack_per_batch: int


@dataclass
class WhitePixelAttack:
    x_dim: Union[int, float]
    y_dim: Union[int, float]
    rest: int
    max_iter: int
    attack_per_batch: int
    pixels_per_iteration: Union[int, float]
    mode: str
    name: str

@dataclass
class Attacks:
    name: str
    type: str


@dataclass
class NIRConfig:
    paths: Paths
    files: Files
    params: Params
    train_params: TrainParams
    model: Models
    params_attack: WhitePixelAttack
    test_params_attack: WhitePixelAttack
    dataset: Data
    attack_type: Attacks
