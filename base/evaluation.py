import json
import logging
import os
import sys
import time
from collections import defaultdict
from itertools import chain
from typing import Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchattacks import OnePixel
from torchvision.transforms import transforms
from tqdm import tqdm

from attacks.base import get_default_attack_config, get_attack, IndexedDataset
from base.corrupted_cifar10 import CorruptedCifar10, CorruptedCifar100, \
    BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS


class NpEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, np.generic):
            return object.item()
        return super(NpEncoder, self).default(object)


def ece_score(ground_truth: Sequence,
              predictions: Sequence,
              probs: Sequence,
              bins: int = 30):
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    probs = np.asarray(probs)

    probs = np.max(probs, -1)

    prob_pred = np.zeros((0,))
    prob_true = np.zeros((0,))
    ece = 0

    mce = []

    for b in range(1, int(bins) + 1):
        i = np.logical_and(probs <= b / bins, probs > (
                b - 1) / bins)  # indexes for p in the current bin

        s = np.sum(i)

        if s == 0:
            prob_pred = np.hstack((prob_pred, 0))
            prob_true = np.hstack((prob_true, 0))
            continue

        m = 1 / s
        acc = m * np.sum(predictions[i] == ground_truth[i])
        conf = np.mean(probs[i])

        prob_pred = np.hstack((prob_pred, conf))
        prob_true = np.hstack((prob_true, acc))
        diff = np.abs(acc - conf)

        mce.append(diff)

        ece += (s / len(ground_truth)) * diff

    return ece, prob_pred, prob_true, mce


@torch.no_grad()
def model_evaluation(model: nn.Module,
                     dataloader: DataLoader):
    device = next(model.parameters()).device

    model.eval()
    tot = 0
    correct = 0

    for i, (inputs, labels) in enumerate(dataloader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        tot += len(outputs)
        correct += torch.argmax(outputs, -1).eq(labels).cpu().sum().item()

    return tot, correct


@torch.no_grad()
def model_evaluator(backbone: nn.Module,
                    classifier: nn.Module,
                    dataloader: DataLoader,
                    device: str = 'cpu'):
    backbone.to(device)
    classifier.to(device)

    backbone.eval()
    classifier.eval()

    total = 0
    correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        e = backbone(inputs)
        outputs = classifier(e)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    score = correct / total

    return score, total, correct


def corrupted_cifar_scores(model, batch_size, use_extra_corruptions=False,
                           dataset='cifar10', normalize=False):
    assert dataset in ['cifar100', 'cifar10']

    if dataset == 'cifar10':
        dataset = CorruptedCifar10
        t = [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010))]
    else:
        dataset = CorruptedCifar100
        t = [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))]

    if use_extra_corruptions:
        corruptions = chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS)
    else:
        corruptions = BENCHMARK_CORRUPTIONS

    scores = {name: {} for name in corruptions}

    for name in corruptions:
        for severity in range(1, 6):
            # CIFAR 10
            loader = torch.utils.data.DataLoader(dataset=
                                                 dataset('~/datasets/',
                                                         download=True,
                                                         corruption=name,
                                                         severity=severity,
                                                         transform=
                                                         transforms.Compose(t)),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True, num_workers=4)

            score = model_evaluation(model, loader)

            scores[name][severity] = score

    return scores


def attack_dataset(model: nn.Module,
                   dataset,
                   attacks,
                   saving_path,
                   images_to_attack_per_label=-1,
                   serialize_names=None,
                   n_classes=None,
                   build_dataset=True):

    if serialize_names is None:
        serialize_names = {}

    log = logging.getLogger(__name__)

    device = next(model.parameters()).device
    model.eval()

    if build_dataset:
        if not isinstance(dataset, IndexedDataset):
            dataset = IndexedDataset(dataset)

        images = []
        labels = []
        indexes = []

        counter = defaultdict(int)

        with torch.no_grad():
            for img, y, image_index in tqdm(DataLoader(dataset,
                                                       batch_size=256,
                                                       shuffle=False),
                                            leave=False):
                img = img.to(device)

                _y = model(img)
                predictions = torch.argmax(_y, -1)

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

    # atks = cfg.get('attacks', {})
    # atks = OmegaConf.to_container(atks)

    all_attack_results = {}

    for k, v in attacks.items():

        v = get_default_attack_config(v)
        attack_factory = get_attack(v)

        log.info('Attack {}, Parameters {}'.format(k, v))

        attack_save_path = os.path.join(saving_path, 'attacks',
                                        '{}.json'.format(serialize_names.get(k, k)))

        attack_images_save_path = os.path.join(saving_path, 'attacks',
                                               '{}_images'.format(
                                                   serialize_names.get(k, k)))

        base_images_save_path = os.path.join(saving_path, 'attacks',
                                             'attacked_images')

        os.makedirs(attack_images_save_path, exist_ok=True)
        os.makedirs(base_images_save_path, exist_ok=True)

        if os.path.exists(attack_save_path) \
                and v.get('load', True):
            try:
                with open(attack_save_path, 'r') \
                        as json_file:
                    attack_results = json.load(json_file)

            except Exception as e:
                attack_results = {'name': k,
                                  'values': v}
        else:
            attack_results = {'name': k,
                              'values': v}

        attack = attack_factory(model)

        if isinstance(attack, OnePixel):
            attack._supported_mode = ['default', 'targeted']

        if len(attack_results) - 2 != len(dataset):
            for img, y, image_index in tqdm(DataLoader(dataset,
                                                       batch_size=1,
                                                       shuffle=False),
                                            leave=False):

                img = img.to(device)
                probs = torch.softmax(model(img), dim=1)[0].tolist()
                image_index = str(image_index.item())

                if image_index in attack_results:
                    d = attack_results[image_index]
                    continue
                else:
                    d = {'label': y.item(),
                         'correct_label_prob': probs[y.item()],
                         'attacks': {}}

                for offset in tqdm(range(n_classes), leave=False):
                    with HiddenPrints():
                        if offset > 0 and v.get('targeted', False):
                            # break
                            f = lambda x, label: (label + offset) % n_classes
                            attack.set_mode_targeted_by_function(f)
                            attack_label = f(img, y)
                            if attack_label == y:
                                attack.set_mode_default()

                        else:
                            offset = 0
                            attack.set_mode_default()
                            attack_label = y

                    attack_label = attack_label.item()

                    if str(attack_label) not in d['attacks']:

                        start = time.time()
                        pert_images = attack(img, y)

                        end = time.time()
                        elapsed_time = end - start

                        iterations = -1
                        statistics = None

                        if getattr(attack, 'probs_statistics', None) \
                                is not None:
                            statistics = attack.probs_statistics[0]
                        if getattr(attack, 'required_iterations', None) \
                                is not None:
                            iterations = attack.required_iterations[0]

                        if v.get('save_images', False):
                            os.makedirs(attack_images_save_path,
                                        exist_ok=True)
                            f = plt.figure()

                            _to_plot = pert_images.cpu().numpy()[0]
                            mn, mx = np.min(_to_plot), \
                                     np.max(_to_plot)

                            _to_plot = (_to_plot - mn) / (mx - mn)
                            _to_plot = np.moveaxis(_to_plot, 0, -1)

                            plt.imshow(_to_plot)
                            plt.axis('off')
                            f.savefig((os.path.join(attack_images_save_path,
                                                    'p_{}_{}'.format(
                                                        image_index, offset))),
                                      bbox_inches='tight')
                            plt.close(f)
                            f = plt.figure()

                            _to_plot = img.detach().cpu().numpy()[0]
                            mn, mx = np.min(_to_plot), \
                                     np.max(_to_plot)

                            _to_plot = (_to_plot - mn) / (mx - mn)
                            _to_plot = np.moveaxis(_to_plot, 0, -1)

                            plt.imshow(_to_plot)
                            plt.axis('off')
                            f.savefig((os.path.join(base_images_save_path,
                                                    'p_{}_{}'.format(
                                                        image_index, offset))),
                                      bbox_inches='tight')
                            plt.close(f)

                        final_prob = torch.softmax(model(pert_images), dim=1)[0]
                        final_prob = final_prob.tolist()

                        diff = (pert_images - img).view(-1)

                        norms = {str(norm_t): torch.linalg.norm(diff,
                                                                ord=norm_t).item() / 3

                                 for norm_t in [0, 2, float('inf')]}

                        res = {
                            # 'attacked_label': attack_label,
                            'time': elapsed_time,
                            # 'probs': final_prob,
                            'correct_label_prob': final_prob[y.item()],
                            'attacked_label_prob': final_prob[attack_label],
                            'highest_label_prob': max(final_prob),

                            'prediction': np.argmax(final_prob),
                            'norms': norms,
                            'iterations': iterations,
                            'statistics': statistics}

                        d['attacks'][str(attack_label)] = res

                attack_results[image_index] = d

                with open(attack_save_path, 'w') \
                        as json_file:
                    json.dump(attack_results, json_file,
                              indent=4, cls=NpEncoder)

        all_attack_results[k] = attack_results

        log.info(f'Results from file {serialize_names.get(k, k)}')

        correctly_attacked, mean_time, std_time, \
        mean_iterations, std_iterations, mean_zeron, std_zeron = \
            calculate_scores(attack_results)

        log.info('\t\tCorrectly attacked: {}/{} ({})'
                 .format(int(correctly_attacked * len(dataset)),
                         len(dataset), correctly_attacked))

        log.info('\t\tIterations required per image: {}(+-{})'
                 .format(mean_iterations, std_iterations))

        log.info('\t\tZero norm per image: {}(+-{})'
                 .format(mean_zeron, std_zeron))

        log.info('\t\tTime required per image: {}(+-{})'
                 .format(mean_time, std_time))

    return all_attack_results


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def calculate_scores(results: dict):
    times = []
    zero_norms = []
    iterations = []

    corrects = 0
    total = 0

    for key in results.keys():
        if key in ['values', 'name']:
            continue

        item = results[key]

        label = item['label']
        res = item['attacks'][str(label)]

        pred = res['prediction']
        times.append(res['time'])
        iterations.append(res['iterations'])
        zero_norms.append(res['norms']['0'])
        total += 1
        if label != pred:
            corrects += 1

    mean_time, std_time = np.mean(times), np.std(times)
    mean_iterations, std_iterations = np.mean(iterations), np.std(iterations)
    mean_zeron, std_zeron = np.mean(zero_norms), np.std(zero_norms)

    correctly_attacked = corrects / total

    return correctly_attacked, \
           mean_time, std_time, \
           mean_iterations, std_iterations, \
           mean_zeron, std_zeron
