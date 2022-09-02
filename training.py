import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from tqdm import tqdm

from resnet import resnet20
from attacks.white_pixle import WhitePixle, PatchWhitePixle


def test(model, loader):
    device = next(model.parameters()).device

    # Accuracy counter
    total = 0
    correct = 0
    adv_examples = []

    model.eval()
    # Loop over all examples in test set
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    final_acc = correct / total
    print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def attack_dataset(model, attack, loader):
    total = 0
    corrects = 0
    device = next(model.parameters()).device
    norms = []
    model.eval()

    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.argmax(1)

        mask = init_pred == target

        data = data[mask]
        target = target[mask]

        if len(data) == 0:
            continue

        adv_images = attack(data, target)

        diff = (adv_images - data).view(data.shape[0], -1)
        l1_norm = torch.linalg.norm(diff, ord=0, dim=-1) / 3
        norms.extend(l1_norm.detach().cpu().numpy())

        output = model(adv_images)
        pred = output.argmax(1)
        corrects += (pred == target).sum().item()
        total += pred.shape[0]

        print(np.mean(norms), total, corrects)

    return total, corrects, norms



test_dataset = datasets.CIFAR10('./dataset', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))

test_dataset = None

train_dataset = datasets.CIFAR10('./dataset', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor()
                         ]))

train_dataset = ImageFolder('./dataset/nirscene1_ir',
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize((420, 420))]))

n = len(train_dataset)
n_test = int(0.1 * n)
idxs = np.arange(n)
np.random.shuffle(idxs)

test_dataset = torch.utils.data.Subset(train_dataset, idxs[:n_test])
train_dataset = torch.utils.data.Subset(train_dataset, idxs[n_test:])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=8, shuffle=True)

if test_dataset is not None:
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8)
else:
    test_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=8)

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Net().to(device)
model = resnet18(num_classes=9).to(device)
print(model.__class__.__name__)

model_save_path = 'data/nir_rgb_model.pth'

if os.path.isfile(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    opt = Adam(model.parameters(), 0.001)

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        test(model, test_loader)

    torch.save(model.state_dict(), model_save_path)

test(model, test_loader)

attack = WhitePixle(attack_limit=100,
                    average_channels=True,
                    model=model,
                    descending=True)

attack = PatchWhitePixle(model=model,
                         x_dimensions=4,
                         y_dimensions=4,
                         restarts=100,
                         max_iterations=100)

attacked, correctly_classified, norms = attack_dataset(model,
                                                       attack,
                                                       test_loader)
print(attacked, correctly_classified, norms)
