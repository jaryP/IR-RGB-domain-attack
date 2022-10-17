import json
import os
from collections import defaultdict
from copy import deepcopy

import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging

from tqdm import tqdm

from attacks import Pixle
from attacks.base import IndexedDataset, get_default_attack_config, get_attack
from attacks.white_pixle import RandomWhitePixle
from base.adv import adv_train
from base.utils import get_model, get_dataset, model_training
from base.evaluation import model_evaluation, corrupted_cifar_scores, \
    attack_dataset
from configs.nir_config import NIRConfig


def test_transfer_dataset(rgb_model,
                          ir_model,
                          attack,
                          rgb_datasets,
                          ir_datasets,
                          source_modal='rgb'):

    assert source_modal in ['ir', 'rgb']
    source_total = 0
    source_attacked = 0

    target_total = 0
    target_attacked = 0

    device = next(rgb_model.parameters()).device

    source_model = rgb_model if source_modal == 'rgb' else ir_model
    target_model = ir_model if source_modal == 'rgb' else rgb_model

    attack.model = source_model

    source_model.eval()
    target_model.eval()

    for i in range(len(rgb_datasets)):
        rgb_image, y = rgb_datasets[i]
        ir_image, _y = ir_datasets[i]

        source_x = rgb_image if source_modal == 'rgb' else ir_image
        target_x = ir_image if source_modal == 'rgb' else rgb_image

        source_x = source_x[None].to(device)
        target_x = target_x[None].to(device)
        target = torch.tensor([y], device=device)

        mask = torch.logical_and(source_model(source_x).argmax(1) == target,
                                 target_model(target_x).argmax(1) == target)

        source_x = source_x[mask]
        target_x = target_x[mask]
        target = target[mask]

        if len(source_x) == 0:
            continue

        target_total += target.shape[0]
        source_total += target.shape[0]

        adv_source_images, solutions = attack(source_x, target, True)

        adv_target_images = target_x.clone()

        solution = solutions[0]

        if solution[0] is None:
            continue

        for v in solution:
            if v is not None:
                aa, bb = v
            else:
                break
            adv_target_images[0, :, aa[:, 0], aa[:, 1]] = \
                adv_target_images[0, :, bb[:, 0], bb[:, 1]]

        source_attacked += (source_model(adv_source_images).argmax(
            1) != target).sum().item()

        target_attacked += (target_model(adv_target_images).argmax(
            1) != target).sum().item()

    return source_total, source_attacked, target_total, target_attacked

logger = logging.getLogger(__name__)

logging.basicConfig(force=True,
                    handlers=[
                        logging.FileHandler("info.log", mode='w'),
                        logging.StreamHandler()
                    ],
                    level=logging.INFO,
                    datefmt='%d/%b/%Y %H:%M:%S',
                    format="[%(asctime)s] %(levelname)s "
                           "[%(name)s:%(lineno)s] %(message)s")

cs = ConfigStore.instance()
cs.store(name="NIR_config", node=NIRConfig)


@hydra.main(config_path="configs", config_name="base_adv_config")
def main(cfg: NIRConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info(OmegaConf.to_yaml(cfg))

    device = 'cpu'
    if cfg.train_params.device != 'cpu' and torch.cuda.is_available():
        device = f'cuda:{cfg.train_params.device}'

    assert 'nir' in cfg.dataset.name.lower()

    models = {}
    datasets = {}

    for dataset_mode, dataset_name in [('ir', 'ir_nir'), ('rgb', 'rgb_nir')]:
        train_set, test_set, input_size, classes = get_dataset(
            name=dataset_name, model_name=None,
            path=cfg.dataset.path,
            resize_dimensions=cfg.params.img_size)

        train, val, test = cfg.params.train, cfg.params.val, cfg.params.test

        if val != 0:
            if isinstance(val, float):
                val = int(val * len(train_set))

            train = len(train_set) - val

            train_set, val_set = torch.utils.data.random_split(train_set, [
                train, val])

            val_loader = torch.utils.data.DataLoader(val_set,
                                                     batch_size=cfg.params.batch_size,
                                                     shuffle=False,
                                                     num_workers=0)
        else:
            val_loader = None

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=cfg.params.batch_size,
                                                   shuffle=True, num_workers=0)

        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=cfg.params.batch_size,
                                                  shuffle=False, num_workers=0)

        current_model = get_model(cfg.model.name, input_size, classes,
                                  pre_trained=getattr(cfg.train_params,
                                                      'pre_trained', False))

        current_model.to(device)

        base_model_path = f'~/leonardo/' \
                          f'{cfg.model.name}/' \
                          f'{dataset_name}/' \
                          f'{cfg.train_params.pre_training_epochs}'

        base_model_path = os.path.expanduser(base_model_path)
        model_path = os.path.join(base_model_path, 'model.pt')

        os.makedirs(base_model_path, exist_ok=True)

        if os.path.exists(model_path):
            logger.info(f'Model {model_path} loaded.')

            current_model.load_state_dict(
                torch.load(os.path.join(base_model_path, 'model.pt'),
                           map_location=device))
        else:
            optimizer = optim.SGD(current_model.parameters(),
                                  lr=cfg.train_params.lr, momentum=0.9)

            current_model = model_training(model=current_model,
                                           epochs=cfg.train_params.pre_training_epochs,
                                           optimizer=optimizer,
                                           dataloader=train_loader)

            torch.save(current_model.state_dict(), model_path)

        if dataset_mode == 'ir':
            pass

        models[dataset_mode] = current_model
        datasets[dataset_mode] = test_set

        current_model.eval()

        tot, corrects = model_evaluation(model=current_model,
                                         dataloader=train_loader)
        logger.info(f'Model trained on {dataset_name}'
                    f' train score {corrects / tot}, '
                    f'({corrects}/{tot})')

        tot, corrects = model_evaluation(model=current_model,
                                         dataloader=test_loader)
        logger.info(f'Model trained on {dataset_name}'
                    f' test score {corrects / tot}, '
                    f'({corrects}/{tot})')

    if hasattr(cfg, 'attacks'):
        atks = cfg.attacks
        atks = OmegaConf.to_container(atks)
    else:
        atks = {}

    for k, v in atks.items():

        v = get_default_attack_config(v)
        attack_factory = get_attack(v)

        logger.info('Attack {}, Parameters {}'.format(k, v))


        attack = attack_factory(models['rgb'])

        source_total, source_attacked, target_total, target_attacked = test_transfer_dataset(
            rgb_model=models['rgb'],
            ir_model=models['ir'],
            ir_datasets=datasets['ir'],
            rgb_datasets=datasets['rgb'],
            attack=attack,
            source_modal='ir')

        logger.info(f'Attacked images {source_total}.\n'
                    f'\tCorrectly attacked IR {source_attacked / source_total}.\n'
                    f'\tCorrectly attacked IR -> RGB {target_attacked / source_total}')


        attack = attack_factory(models['ir'])

        source_total, source_attacked, target_total, target_attacked = test_transfer_dataset(
            rgb_model=models['rgb'],
            ir_model=models['ir'],
            ir_datasets=datasets['ir'],
            rgb_datasets=datasets['rgb'],
            attack=attack,
            source_modal='rgb')

        logger.info(f'Attacked images {source_total}.\n'
                    f'\tCorrectly attacked RGB {source_attacked / source_total}.\n'
                    f'\tCorrectly attacked RGB -> IR {target_attacked / source_total}')

        logger.info('The training process is over')


if __name__ == '__main__':
    main()
