import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging

from attacks import Pixle, RandomWhitePixle
from base.adv import adv_train, adv_testing, adv_validation
from base.utils import get_model, get_dataset
from configs.nir_config import NIRConfig

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="NIR_config", node=NIRConfig)


@hydra.main(config_path="configs", config_name="base_adv_config")
def main(cfg: NIRConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info(OmegaConf.to_yaml(cfg))

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

    if cfg.attack_type.name == 'white':
        attack = RandomWhitePixle(model=model,
                                  x_dimensions=cfg.params_attack.x_dim,
                                  y_dimensions=cfg.params_attack.y_dim,
                                  restarts=cfg.params_attack.rest,
                                  max_iterations=cfg.params_attack.max_iter)

    elif cfg.attack_type.name == 'black':
        attack = Pixle(model=model,
                       x_dimensions=cfg.params_attack.x_dim,
                       y_dimensions=cfg.params_attack.y_dim,
                       restarts=cfg.params_attack.rest,
                       max_iterations=cfg.params_attack.max_iter)
    else:
        assert False, 'The possible attacks are: white, black'

    if os.path.exists('model.pt'):
        logger.info('Model loaded.')
        model.load_state_dict(os.path.join(os.getcwd(), 'model.pt'),
                              map_location=device)
    else:
        scores = {'train': {}, 'dev': {}, 'test': None}

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.train_params.lr)

        for epoch in range(cfg.train_params.epoch_count):
            adv_train(epoch, train_loader, model, optimizer, loss,
                      cfg.train_params.log_freq,
                      attack=attack,
                      type_of_attack=cfg.attack_type.name,
                      attack_per_batch=cfg.params_attack.attack_per_batch)

            # adv_validation(model, val_loader, loss, attack=attack,
            #                type_of_attack=cfg.attack_type.name,
            #                attack_per_batch=cfg.params_attack.attack_per_batch)

            dev_scores = adv_testing(model, val_loader, attack=attack,
                                     type_of_attack=cfg.attack_type.name,
                                     attack_per_batch=cfg.params_attack.attack_per_batch)

            train_scores = adv_testing(model, train_loader, attack=attack,
                                       type_of_attack=cfg.attack_type.name,
                                       attack_per_batch=cfg.params_attack.attack_per_batch)

            scores['train'][epoch] = train_scores
            scores['dev'][epoch] = dev_scores

        test_scores = adv_testing(model, test_loader, attack=attack,
                                  type_of_attack=cfg.attack_type.name,
                                  attack_per_batch=cfg.params_attack.attack_per_batch)

        scores['test'] = test_scores

        with open(os.path.join(os.getcwd(), 'results.pt'), 'wb') as file:
            pickle.dump(scores, file)

        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model.pt'))

    with open(os.path.join(os.getcwd(), 'results.pt'), 'rb') as file:
        results = pickle.load(file)
        train_scores, dev_scores, test_scores = results['train'], \
                                                results['dev'],\
                                                results['test']

    for k in train_scores:
        logger.info(f'Epoch {k}')
        logger.info(f'Train scores {train_scores[k]}')
        logger.info(f'Dev scores {dev_scores[k]}')

    logger.info(f'Final test scores {test_scores}')

    # test_Scores = adv_testing(model, test_loader, attack=attack,
    #                         type_of_attack=cfg.attack_type.name,
    #                         attack_per_batch=cfg.params_attack.attack_per_batch)


if __name__ == '__main__':
    main()
