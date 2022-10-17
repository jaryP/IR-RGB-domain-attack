import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from loss_landscapes.model_interface.model_parameters import ModelParameters
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging

from tqdm import tqdm

from attacks import Pixle, PatchWhitePixle
from attacks.base import IndexedDataset
from attacks.white_pixle import RandomWhitePixle
from base.adv import adv_train, adv_testing, adv_validation
from base.utils import get_model, get_dataset, model_training
from base.evaluation import model_evaluation, corrupted_cifar_scores, \
    attack_dataset
from configs.nir_config import NIRConfig
import loss_landscapes

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

# handler = logging.FileHandler('file.log', mode='w')
# logger.addHandler(handler)

# logger = logging.basicConfig(filemode=logging.DEBUG, filemode='w',
#                              filename='file.log',
#                              format='%(asctime)s %(levelname)s % (message)s')
#
# def rand_u_like(example_vector: ModelParameters) -> ModelParameters:
#     new_vector = []
#
#     for param in example_vector:
#         new_vector.append(torch.rand(size=param.size(), dtype=example_vector[0].dtype).to(param.device))
#
#     return ModelParameters(new_vector)
#
#
# def rand_n_like(example_vector: ModelParameters) -> ModelParameters:
#     new_vector = []
#
#     for param in example_vector:
#         new_vector.append(torch.randn(size=param.size(), dtype=example_vector[0].dtype).to(param.device))
#
#     return ModelParameters(new_vector)


cs = ConfigStore.instance()
cs.store(name="NIR_config", node=NIRConfig)


@hydra.main(config_path="configs", config_name="base_adv_config")
def main(cfg: NIRConfig):
    print(OmegaConf.to_yaml(cfg))

    images_to_attack_per_label = cfg.params.images_to_attack_per_label

    logger.info(OmegaConf.to_yaml(cfg))

    device = 'cpu'
    if cfg.train_params.device != 'cpu' and torch.cuda.is_available():
        device = f'cuda:{cfg.train_params.device}'

    train_set, test_set, input_size, classes = get_dataset(
        name=cfg.dataset.name, model_name=None,
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
                                                 shuffle=False, num_workers=0)
    else:
        val_loader = None

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.params.batch_size,
                                               shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=cfg.params.batch_size,
                                              shuffle=False, num_workers=0)

    pre_trained_model = get_model(cfg.model.name, input_size, classes,
                                  pre_trained=getattr(cfg.train_params,
                                                      'pre_trained', False))

    if val_loader is None:
        val_str = 'test'
        val_loader = test_loader
    else:
        val_str = 'dev'

    pre_trained_model.to(device)
    adv_trained_model = deepcopy(pre_trained_model)

    if 'white' in cfg.test_params_attack.name:
        attack = RandomWhitePixle(model=adv_trained_model,
                                  mode=cfg.params_attack.mode,
                                  pixels_per_iteration=cfg.params_attack.pixels_per_iteration,
                                  restarts=cfg.params_attack.rest,
                                  max_iterations=cfg.params_attack.max_iter)

    elif 'black' in cfg.test_params_attack.name:
        attack = Pixle(model=adv_trained_model,
                       x_dimensions=cfg.params_attack.x_dim,
                       y_dimensions=cfg.params_attack.y_dim,
                       restarts=cfg.params_attack.rest,
                       max_iterations=cfg.params_attack.max_iter)
    else:
        assert False, 'The possible attacks are: white, black'

    # if 'white' in cfg.test_params_attack.name:
    #     test_attack = RandomWhitePixle(model=model,
    #                                    mode=cfg.test_params_attack.mode,
    #                                    pixels_per_iteration=cfg.test_params_attack.pixels_per_iteration,
    #                                    restarts=cfg.test_params_attack.rest,
    #                                    iterations=cfg.test_params_attack.max_iter)
    #
    # elif 'black' in cfg.test_params_attack.name:
    #     test_attack = Pixle(model=model,
    #                         x_dimensions=cfg.test_params_attack.x_dim,
    #                         y_dimensions=cfg.test_params_attack.y_dim,
    #                         restarts=cfg.test_params_attack.rest,
    #                         max_iterations=cfg.test_params_attack.max_iter)
    # else:
    #     assert False, 'The possible attacks are: white, black, {} given'

    # attacks_dict = {
    #     'test_attack': OmegaConf.to_container(cfg.test_params_attack)}

    if not os.path.exists('results.pt'):
        base_model_path = f'~/leonardo/' \
                          f'{cfg.model.name}/' \
                          f'{cfg.dataset.name}/' \
                          f'{cfg.train_params.pre_training_epochs}'

        base_model_path = os.path.expanduser(base_model_path)

        os.makedirs(base_model_path, exist_ok=True)

        if os.path.exists('adv_model.pt'):
            logger.info('Adv. Model loaded.')

            pre_trained_model.load_state_dict(
                torch.load(os.path.join(base_model_path, 'model.pt'),
                           map_location=device))

            adv_trained_model.load_state_dict(
                torch.load('adv_model.pt',
                           map_location=device))

            with open('results.json', 'r') as file:
                accuracy_scores = json.load(file)

        else:
            attack_scores = {'train': {}, 'dev': {}, 'test': {},
                             'corrupted': {}}
            accuracy_scores = {'train': [], 'dev': [], 'test': [],
                               'corrupted': []}

            loss = nn.CrossEntropyLoss()

            if os.path.exists(os.path.join(base_model_path, 'model.pt')):
                logger.info('Model loaded.')
                pre_trained_model.load_state_dict(
                    torch.load(os.path.join(base_model_path, 'model.pt'),
                               map_location=device))

            else:
                optimizer = optim.Adam(pre_trained_model.parameters(),
                                       lr=cfg.train_params.lr)

                pre_trained_model = model_training(model=pre_trained_model,
                                                   epochs=cfg.train_params.pre_training_epochs,
                                                   optimizer=optimizer,
                                                   dataloader=train_loader)

                torch.save(pre_trained_model.state_dict(),
                           os.path.join(base_model_path, 'model.pt'))

            # attack_dataset(model=pre_trained_model,
            #                dataset=test_set,
            #                saving_path='.',
            #                images_to_attack_per_label=images_to_attack_per_label,
            #                attacks=attacks_dict,
            #                serialize_names={
            #                    'test_attack': 'pretrained_test_attack'})

            tot, corrects = model_evaluation(model=pre_trained_model,
                                             dataloader=test_loader)
            accuracy_scores['train'].append((tot, corrects))
            logger.info(
                f'base model score: {corrects / tot} ({corrects}/{tot}).')

            tot, corrects = model_evaluation(model=pre_trained_model,
                                             dataloader=test_loader)
            accuracy_scores['test'].append((tot, corrects))

            if val_str == 'dev':
                tot, corrects = model_evaluation(model=pre_trained_model,
                                                 dataloader=val_loader)
                accuracy_scores[val_str].append((tot, corrects))

            adv_trained_model.load_state_dict(pre_trained_model.state_dict())
            optimizer = optim.Adam(adv_trained_model.parameters(),
                                   lr=cfg.train_params.lr)

            # dev_scores = adv_testing(model, val_loader, attack=test_attack)
            # total, attacked_images, correctly_attacked = dev_scores
            # logger.info(
            #     f'(pre-training results) Dev Images {total}, attacked images {attacked_images}')
            # logger.info(f'Correctly attacked images: {correctly_attacked}')
            # attack_scores[val_str]['pretraining'] = dev_scores

            if cfg.dataset.name == 'cifar10':
                ccifar10_results = corrupted_cifar_scores(adv_trained_model,
                                                          batch_size=32,
                                                          dataset=cfg.dataset.name)

                # for corruption, severity in ccifar10_results.items():
                #     for s, v in severity.items():
                #         logger.info(f'Corruption {corruption} with severity '
                #                     f'{s}. Score: {(v[1] / v[0]) * 100}.')

                accuracy_scores['corrupted'].append(ccifar10_results)

            for epoch in tqdm(range(cfg.train_params.adv_training_epochs)):
                losses, correct, total = adv_train(epoch=epoch,
                                                   loader=train_loader,
                                                   model=adv_trained_model,
                                                   optimizer=optimizer,
                                                   loss_func=loss,
                                                   log_freq=cfg.train_params.log_freq,
                                                   attack=attack,
                                                   type_of_attack=cfg.params_attack.name,
                                                   attack_per_batch=cfg.params_attack.attack_per_batch)

                logger.info(
                    f'Epoch {epoch} (corrupted) train score {correct / total}, '
                    f'({correct}/{total})')

                tot, corrects = model_evaluation(model=adv_trained_model,
                                                 dataloader=train_loader)
                accuracy_scores['train'].append((tot, corrects))

                logger.info(
                    f'Epoch {epoch} train score {corrects / tot}, '
                    f'({corrects}/{tot})')

                tot, corrects = model_evaluation(model=adv_trained_model,
                                                 dataloader=test_loader)
                accuracy_scores['test'].append((tot, corrects))

                if val_str == 'dev':
                    tot, corrects = model_evaluation(model=adv_trained_model,
                                                     dataloader=val_loader)
                    accuracy_scores[val_str].append((tot, corrects))

                if cfg.dataset.name == 'cifar10':
                    ccifar10_results = corrupted_cifar_scores(adv_trained_model,
                                                              batch_size=32,
                                                              dataset=cfg.dataset.name)

                    accuracy_scores['corrupted'].append(ccifar10_results)

                # if (epoch + 1) % cfg.train_params.eval_every_n_epochs == 0 or \
                #         epoch == cfg.train_params.fine_tune_epochs - 1:

                # if cfg.dataset.name == 'cifar10':
                #     ccifar10_results = corrupted_cifar_scores(model,
                #                                               batch_size=32,
                #                                               dataset=cfg.dataset.name)
                #
                #     for corruption, severity in ccifar10_results.items():
                #         for s, v in severity.items():
                #             logger.info(
                #                 f'Corruption {corruption} with severity '
                #                 f'{s}. Score: {(v[1] / v[0]) * 100}.')
                #
                #     attack_scores['corrupted'][epoch] = ccifar10_results
                # else:

                # dev_scores = adv_testing(model, val_loader,
                #                          attack=test_attack)

                # total, attacked_images, correctly_attacked = dev_scores
                # logger.info(
                #     f'Dev Images {total}, attacked images {attacked_images}')
                # logger.info(
                #     f'Correctly attacked images: {correctly_attacked}')

                # train_scores = adv_testing(model, train_loader, attack=attack)
                #
                # total, attacked_images, correctly_attacked = dev_scores
                # logger.info(
                #     f'Dev Images {total}, attacked images {attacked_images}')
                # logger.info(f'Correctly attacked images: {correctly_attacked}')

                # scores['train'][epoch] = train_scores
                # attack_scores[val_str][epoch] = dev_scores

                with open('results_temp.json', 'w') as file:
                    json.dump({"attack": attack_scores,
                               'accuracy': accuracy_scores}, file, indent=4)

            # test_scores = adv_testing(model, test_loader, attack=test_attack)

            # attack_scores['test'] = test_scores

            with open('results.json', 'w') as file:
                json.dump({"attack": attack_scores,
                           'accuracy': accuracy_scores}, file, indent=4)

            torch.save(adv_trained_model.state_dict(), 'adv_model.pt')

        adv_trained_model.eval()
        pre_trained_model.eval()

        images = []
        labels = []
        indexes = []

        counter = defaultdict(int)

        with torch.no_grad():
            for img, y, image_index in tqdm(DataLoader(IndexedDataset(test_set),
                                                       batch_size=256,
                                                       shuffle=False),
                                            leave=False):
                img = img.to(device)

                _y_pre = pre_trained_model(img)
                predictions = torch.argmax(_y_pre, -1)

                mask = torch.argmax(adv_trained_model(img), -1) == predictions

                img = img[mask]
                predictions = predictions[mask]
                y = y[mask]

                for j, (im, p, gt) in enumerate(zip(img.cpu(),
                                                    predictions.cpu().numpy(),
                                                    y.numpy())):

                    if p == gt and (counter[gt] < images_to_attack_per_label
                                    or images_to_attack_per_label < 0):
                        counter[gt] += 1
                        images.append(im)
                        indexes.append(image_index[j])
                        labels.append(gt)

        n_classes = len(counter)

        images = torch.stack(images, 0)
        labels = torch.tensor(labels)
        indexes = torch.tensor(indexes)

        dataset = TensorDataset(images, labels, indexes)

        tot, corrects = model_evaluation(model=pre_trained_model,
                                         dataloader=test_loader)
        logger.info(f'Pre-trained test score {corrects / tot}, '
            f'({corrects}/{tot})')

        tot, corrects = model_evaluation(model=adv_trained_model,
                                         dataloader=test_loader)
        logger.info(f'Adv-trained test score {corrects / tot}, '
            f'({corrects}/{tot})')

        if hasattr(cfg, 'attacks'):
            atks = cfg.attacks
            atks = OmegaConf.to_container(atks)

            attack_dataset(model=pre_trained_model,
                           build_dataset=False,
                           dataset=dataset,
                           n_classes=n_classes,
                           saving_path='.',
                           images_to_attack_per_label=images_to_attack_per_label,
                           attacks=atks,
                           serialize_names=lambda x: 'pretrained_'+x)

            attack_dataset(model=adv_trained_model,
                           build_dataset=False,
                           dataset=dataset,
                           n_classes=n_classes,
                           saving_path='.',
                           images_to_attack_per_label=images_to_attack_per_label,
                           attacks=atks,
                           serialize_names=lambda x: 'final_'+x)

    if cfg.dataset.name == 'cifar10':
        for k, severity in accuracy_scores['accuracy']['corrupted'][-1].items():
            v = 0

            for si, (tot, pred) in severity.items():
                otot, opred = accuracy_scores['accuracy']['corrupted'][0][k][si]
                v += (pred / tot) / (opred / otot)

            logger.info(f'Corruption {k}, CE: {v / 5}')

    # x, y = next(iter(DataLoader(test_set, batch_size=512)))
    # # x, y = x.to(device), y.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    # metric = loss_landscapes.metrics.Loss(criterion, x, y)
    # loss_data_fin = loss_landscapes.random_plane(pre_trained_model.cpu(), metric, 10, 10,
    #                                              normalization='filter',
    #                                              deepcopy_model=True)
    #
    # plt.figure()
    # plt.contour(loss_data_fin, levels=50)
    # plt.title('Loss Contours around Trained Model')
    # plt.savefig('pre_cont.pdf')
    # plt.close()
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # X = np.array([[j for j in range(10)] for i in range(10)])
    # Y = np.array([[i for _ in range(10)] for i in range(10)])
    # ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax.set_title('Surface Plot of Loss Landscape')
    # plt.savefig('pre_surf.pdf')
    #
    # loss_data_fin = loss_landscapes.random_plane(adv_trained_model.cpu(), metric, 10,
    #                                              10,
    #                                              normalization='filter',
    #                                              deepcopy_model=True)
    #
    # plt.figure()
    # plt.contour(loss_data_fin, levels=50)
    # plt.title('Loss Contours around Trained Model')
    # plt.savefig('post_cont.pdf')
    # plt.close()
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # X = np.array([[j for j in range(10)] for i in range(10)])
    # Y = np.array([[i for _ in range(10)] for i in range(10)])
    # ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax.set_title('Surface Plot of Loss Landscape')
    # plt.savefig('post_surf.pdf')

    logger.info('The training process is over')


if __name__ == '__main__':
    main()
