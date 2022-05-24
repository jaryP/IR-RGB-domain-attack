import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet import resnet20
from attacks.white_pixle import WhitePixle, RandomWhitePixle


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

    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.argmax(1)

        mask = init_pred == target

        data = data[mask]
        target = target[mask]

        adv_images = attack(data, target)

        diff = (adv_images - data).view(data.shape[0], -1)
        l1_norm = torch.linalg.norm(diff, ord=0, dim=-1) / 3
        norms.extend(l1_norm.detach().cpu().numpy())

        print(np.mean(norms))
        output = model(adv_images)
        pred = output.argmax(1)
        corrects += (pred == target).sum().item()
        total += pred.shape[0]

    return total, corrects, norms


test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./dataset', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ])),
    batch_size=32, shuffle=False)

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Net().to(device)
model = resnet20().to(device)
print(model.__class__.__name__)

if os.path.isfile('data/model.pth'):
    model.load_state_dict(torch.load('data/model.pth', map_location=device))
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./dataset', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ])),
        batch_size=32, shuffle=True)

    opt = Adam(model.parameters(), 0.001)
    
    for epoch in range(10):
        if epoch > 0:
            attack = WhitePixle(attack_limit=epoch * 2, average_channels=True, 
                                model=model)
        else:
            attack = None
        
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if attack is not None:
                x = attack(x, y)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        test(model, test_loader)

    torch.save(model.state_dict(), 'data/model.pth')

test(model, test_loader)

attack = WhitePixle(attack_limit=100,
                    average_channels=True,
                    model=model,
                    descending=False)

attack = RandomWhitePixle(model=model,
                          x_dimensions=1,
                          y_dimensions=1,
                          restarts=50,
                          max_iterations=10)

attacked, correctly_classified, norms = attack_dataset(model,
                                                       attack,
                                                       test_loader)
print(attacked, correctly_classified, norms)
