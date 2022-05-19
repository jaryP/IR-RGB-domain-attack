import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet import resnet20
from attacks.white_pixle import WhitePixle


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

# total = 0
# corrects = 0
#
# for data, target in tqdm(test_loader):
#     data = data.to(device)
#     target = target.to(device)
#
#     adv_images = attack(data, target)
#
#     output = model(adv_images)
#     pred = output.argmax(1)  # get the index of the max log-probability
#     corrects += (pred == target).sum()
#     total += pred.shape[0]

attacked, correctly_attacked, norms = attack_dataset(model, attack, test_loader)
print(attacked, correctly_attacked)

# modified_pixels = []
# 
# total = 0
# corrects = 0
# 
# model.eval()
# 
# average_channels = True
# swap = False
# 
# for data, target in tqdm(test_loader):
#     data, target = data.to(device), target.to(device)
#     output = model(data)
#     init_pred = output.argmax(1)  # get the index of the max log-probability
# 
#     mask = init_pred == target
# 
#     data = data[mask]
#     target = target[mask]
#     output = output[mask]
# 
#     data.requires_grad = True
# 
#     loss = F.cross_entropy(model(data), target)
# 
#     model.zero_grad()
#     loss.backward()
# 
#     data_grad = data.grad.data
# 
#     if average_channels:
#         data_grad = data_grad.mean(1)
# 
#     data_grad = torch.abs(data_grad)
# 
#     data_grad = torch.flatten(data_grad, 1)
#     indexes = torch.argsort(data_grad, -1)
# 
#     adv_images = []
# 
#     for img_i in range(len(data)):
#         img = data[img_i]
#         adv_img = img.clone()
#         img_target = target[img_i]
# 
#         img_grads = data_grad[img_i]
#         img_indexes = indexes[img_i]
# 
#         for i in range(len(img_indexes) // 2):
#             less_important, more_important = img_indexes[i], img_indexes[
#                 - (i + 1)]
#             if average_channels:
#                 a = np.unravel_index(less_important.item(), (32, 32))
#                 b = np.unravel_index(more_important.item(), (32, 32))
#             else:
#                 a = np.unravel_index(less_important.item(), (3, 32, 32))
#                 b = np.unravel_index(more_important.item(), (3, 32, 32))
# 
#             if average_channels:
#                 # adv_img[:, a] = img[:, b]
#                 v = adv_img[:, b]
#                 adv_img[:, b] = img[:, a]
#                 if swap:
#                     adv_img[:, a] = v
#             else:
#                 v = adv_img[b]
#                 adv_img[b] = img[a]
#                 if swap:
#                     adv_img[a] = v
# 
#             output = model(adv_img[None, :])
#             pred = output.argmax(-1)
# 
#             if img_target.item() != pred.item():
#                 modified_pixels.append(i + 1)
#                 break
# 
#         adv_images.append(adv_img)
# 
#     adv_images = torch.stack(adv_images, 0)
# 
#     output = model(adv_images)
#     pred = output.argmax(1)  # get the index of the max log-probability
#     corrects += (pred == target).sum()
#     total += pred.shape[0]
# 
# print(corrects, total, corrects / total)
#
# print(modified_pixels)
# print(np.mean(modified_pixels))
# print(np.std(modified_pixels))
