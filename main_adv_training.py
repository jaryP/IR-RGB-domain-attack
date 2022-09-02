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

from tqdm import tqdm

from attacks import Pixle, PatchWhitePixle
from attacks.white_pixle import RandomWhitePixle
from base.adv import adv_train, adv_testing, adv_validation
from base.utils import get_model, get_dataset, model_training, model_evaluation
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
                      pre_trained=getattr(cfg.train_params,
                                          'pre_trained', False))

    model.to(device)

    if cfg.attack_type.name == 'white':
        attack = RandomWhitePixle(model=model,
                                  mode=cfg.params_attack.mode,
                                  pixels_per_iteration=cfg.params_attack.pixels_per_iteration,
                                  restarts=cfg.params_attack.rest,
                                  iterations=cfg.params_attack.max_iter)

    elif cfg.attack_type.name == 'black':
        attack = Pixle(model=model,
                       x_dimensions=cfg.params_attack.x_dim,
                       y_dimensions=cfg.params_attack.y_dim,
                       restarts=cfg.params_attack.rest,
                       max_iterations=cfg.params_attack.max_iter)
    else:
        assert False, 'The possible attacks are: white, black'

    base_model_path = f'~/leonardo/{cfg.model.name}/{cfg.dataset.name}'
    base_model_path = os.path.expanduser(base_model_path)

    os.makedirs(base_model_path, exist_ok=True)

    if os.path.exists(os.path.join(base_model_path, 'adv_model.pt')):
        logger.info('Adv. Model loaded.')
        model.load_state_dict(
            torch.load(os.path.join(base_model_path, 'adv_model.pt'),
                       map_location=device))
    else:
        scores = {'train': {}, 'dev': {}, 'test': None}

        loss = nn.CrossEntropyLoss()

        if os.path.exists(os.path.join(base_model_path, 'model.pt')):
            logger.info('Model loaded.')
            model.load_state_dict(
                torch.load(os.path.join(base_model_path, 'model.pt'),
                           map_location=device))
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.train_params.lr)

            model = model_training(model=model,
                                   epochs=cfg.train_params.fine_tune_epochs,
                                   optimizer=optimizer, dataloader=train_loader)

            torch.save(model.state_dict(),
                       os.path.join(base_model_path, 'model.pt'))

        tot, corrects = model_evaluation(model=model, dataloader=test_loader)
        logger.info(f'base model score: {corrects / tot} ({corrects}/{tot}).')

        optimizer = optim.Adam(model.parameters(), lr=cfg.train_params.lr)

        for epoch in tqdm(range(cfg.train_params.fine_tune_epochs)):
            adv_train(epoch=epoch, loader=train_loader, model=model,
                      optimizer=optimizer,
                      loss_func=loss,
                      log_freq=cfg.train_params.log_freq,
                      attack=attack,
                      type_of_attack=cfg.attack_type.name,
                      attack_per_batch=cfg.params_attack.attack_per_batch)

            # adv_validation(model, val_loader, loss, attack=attack,
            #                type_of_attack=cfg.attack_type.name,
            #                attack_per_batch=cfg.params_attack.attack_per_batch)

            dev_scores = adv_testing(model, val_loader, attack=attack)

            total, attacked_images, correctly_attacked = dev_scores
            logger.info(
                f'Dev Images {total}, attacked images {attacked_images}')
            logger.info(f'Correctly attacked images: {correctly_attacked}')

            # train_scores = adv_testing(model, train_loader, attack=attack)
            #
            # total, attacked_images, correctly_attacked = dev_scores
            # logger.info(
            #     f'Dev Images {total}, attacked images {attacked_images}')
            # logger.info(f'Correctly attacked images: {correctly_attacked}')

            # scores['train'][epoch] = train_scores
            scores['dev'][epoch] = dev_scores

        test_scores = adv_testing(model, test_loader, attack=attack)

        scores['test'] = test_scores

        with open(os.path.join(os.getcwd(), 'results.pt'), 'wb') as file:
            pickle.dump(scores, file)

        torch.save(model.state_dict(),
                   os.path.join(base_model_path, 'adv_model.pt'))

    with open(os.path.join(os.getcwd(), 'results.pt'), 'rb') as file:
        results = pickle.load(file)
        train_scores, dev_scores, test_scores = results['train'], \
                                                results['dev'], \
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
