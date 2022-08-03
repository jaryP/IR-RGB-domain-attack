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


@dataclass
class Train_params:
    epoch_count: int
    lr: float
    log_freq: int
    device: Union[str, int]


@dataclass
class Models:
    name: str


@dataclass
class Data:
    path: str
    name: str

@dataclass
class Par_attack:
    x_dim: int
    y_dim: int
    rest: int
    max_iter: int
    attack_per_batch: int


@dataclass
class Attacks:
    name: str
    type: str


@dataclass
class NIRConfig:
    paths: Paths
    files: Files
    params: Params
    train_params: Train_params
    model: Models
    params_attack: Par_attack
    dataset: Data
    attack_type: Attacks
