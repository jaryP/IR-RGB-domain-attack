import logging
import os
import warnings
from itertools import chain

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from attacks.base import IndexedDataset
from evaluators import accuracy_score
from base.utils import get_optimizer, get_dataset, get_model
from base.evaluation import attack_dataset


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)

    logging.basicConfig(force=True,
                        handlers=[
                            logging.FileHandler("info.log", mode='w'),
                            logging.StreamHandler()
                        ],
                        level=logging.INFO,
                        datefmt='%d/%b/%Y %H:%M:%S',
                        format="[%(asctime)s] %(levelname)s "
                               "[%(name)s:%(lineno)s] %(message)s")

    log.info(OmegaConf.to_yaml(cfg))

    experiment_cfg = cfg['experiment']
    load, save, path, experiments = experiment_cfg.get('load', True), \
                                    experiment_cfg.get('save', True), \
                                    experiment_cfg.get('path', None), \
                                    experiment_cfg.get('experiments', 1)

    device = cfg['training'].get('device', 'cpu')

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn("Device not found or CUDA not available.")

    device = torch.device(device)

    if path is None:
        path = os.getcwd()
    else:
        os.chdir(path)
        os.makedirs(path, exist_ok=True)

    for image_index in range(experiments):
        torch.manual_seed(image_index)
        np.random.seed(image_index)

        experiment_path = os.path.join(path, 'exp_{}'.format(image_index))
        os.makedirs(experiment_path, exist_ok=True)

        experiment_cfg = cfg['experiment']
        load, save, path, experiments = experiment_cfg.get('load', True), \
                                        experiment_cfg.get('save', True), \
                                        experiment_cfg.get('path', None), \
                                        experiment_cfg.get('experiments', 1)

        if path is None:
            path = os.getcwd()
        else:
            os.chdir(path)
            os.makedirs(path, exist_ok=True)

        model_cfg = cfg['model']
        model_name = model_cfg['name']
        pre_trained = model_cfg.get('pretrained', False)

        dataset_cfg = cfg['dataset']
        dataset_name = dataset_cfg['name']
        augmented_dataset = dataset_cfg.get('augment', False)
        images_to_attack_per_label = dataset_cfg.get('images_to_attack_'
                                                     'per_label', 50)
        dataset_path = dataset_cfg.get('path', None)

        is_imagenet = dataset_name.lower() == 'imagenet'

        experiment_cfg = cfg['experiment']

        experiments = experiment_cfg.get('experiments', 1)
        plot = experiment_cfg.get('plot', False)

        training_cfg = cfg['training']
        epochs, batch_size = training_cfg['epochs'], \
                             training_cfg['batch_size']
        resize_dimensions = training_cfg.get('resize_dimensions', None)

        optimizer_cfg = cfg['optimizer']
        optimizer_name, lr, momentum, weight_decay = optimizer_cfg.get(
            'optimizer',
            'sgd'), \
                                                     optimizer_cfg.get('lr',
                                                                       1e-1), \
                                                     optimizer_cfg.get(
                                                         'momentum',
                                                         0.9), \
                                                     optimizer_cfg.get(
                                                         'weight_decay', 0)

        os.makedirs(path, exist_ok=True)

        train_set, test_set, input_size, classes = \
            get_dataset(name=dataset_name,
                        resize_dimensions=resize_dimensions,
                        model_name=None,
                        augmentation=augmented_dataset,
                        path=dataset_path)

        test_set = IndexedDataset(test_set)

        if not is_imagenet:
            train_set = IndexedDataset(train_set)
            trainloader = torch.utils.data.DataLoader(train_set,
                                                      batch_size=batch_size,
                                                      shuffle=True)

        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        model = get_model(model_name,
                          image_size=input_size,
                          classes=classes + 1 if not is_imagenet else classes,
                          is_imagenet=is_imagenet,
                          pre_trained=pre_trained)

        model.to(device)

        if not is_imagenet:
            if os.path.exists(os.path.join(experiment_path,
                                           'model.pt')):
                log.info('Model loaded.')

                model.load_state_dict(torch.load(
                    os.path.join(experiment_path, 'model.pt'),
                    map_location=device))

            else:
                log.info('Training model.')

                parameters = chain(model.parameters())

                optimizer = get_optimizer(parameters=parameters,
                                          name=optimizer_name,
                                          lr=lr,
                                          momentum=momentum,
                                          weight_decay=weight_decay)

                model.to(device)

                bar = tqdm(range(epochs))

                for epoch in bar:
                    model.train()

                    for _, (inputs, labels, image_index) in enumerate(
                            trainloader, 0):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)

                        loss = nn.functional.cross_entropy(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()

                    test_score, _, _ = accuracy_score(model=model,
                                                      dataset=testloader,
                                                      device=device)

                    train_score, _, _ = accuracy_score(model=model,
                                                       dataset=trainloader,
                                                       device=device)

                    bar.set_postfix({'train score': train_score,
                                     'test sore': test_score})

                torch.save(model.state_dict(),
                           os.path.join(experiment_path, 'model.pt'))

            with torch.no_grad():
                test_score, _, _ = accuracy_score(model=model,
                                                  dataset=testloader,
                                                  device=device)

                train_score, _, _ = accuracy_score(model=model,
                                                   dataset=trainloader,
                                                   device=device)

                log.info(f"Train score: {train_score}")
                log.info(f"Test score: {test_score}")

        model.eval()

        atks = cfg.get('attacks', {})
        atks = OmegaConf.to_container(atks)

        all_attack_results = attack_dataset(model=model,
                                            dataset=test_set,
                                            attacks=atks,
                                            saving_path=experiment_path,
                                            images_to_attack_per_label=images_to_attack_per_label)

        return model


if __name__ == "__main__":
    my_app()
