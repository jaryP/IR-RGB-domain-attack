import numpy as np
import torch
import random
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def adv_train(epoch, loader, model, optimizer, loss_func, log_freq, attack,
              type_of_attack, attack_per_batch):

    device = next(model.parameters()).device

    # log_freq è il range di batch che aspetto ogni volta che calcolo le metriche
    model.train()
    running_loss = 0
    # numero img classificate correttamente
    correct = 0
    # numero totale img
    total = 0
    losses = []
    # numero di attacchi generati
    adv_num = 0
    # numero di attacchi che hanno avuto successo
    adv_missclass = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, leave=False)):

        inputs, targets = inputs.to(device), targets.to(device)
        # indici delle img da attaccare ogni batch
        # può succedere che l'ultima batch sia più piccola rispetto alle altre e che il numero di attacchi da fare sia superiore
        # perciò setto queste condizione nel caso in cui ci sia questo caso
        # es : ultima batch: 3  --- attacchi per batch : 4 --> ERRORE ---> attacchi per batch : 4/2
        # if len(inputs) < attack_per_batch and len(inputs) > 1:
        #     attack_per_batch = len(inputs) // 2
        #     idx = random.sample(range(len(inputs)), attack_per_batch)
        if isinstance(attack_per_batch, float):
            _attack_per_batch = int(attack_per_batch * len(inputs))
        else:
            _attack_per_batch = attack_per_batch

        if len(inputs) > _attack_per_batch:
            idx = torch.randperm(inputs.shape[0])[:attack_per_batch]
            inputs[idx] = attack(inputs[idx], targets[[idx]])
        else:
            inputs = attack(inputs, targets)  # genera attacco

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_func(outputs, targets)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    # if type_of_attack == 'black':
    #     print('The version of the attack is:', type_of_attack)
    #     for batch_idx, (inputs, targets) in enumerate(tqdm(loader, leave=False)):
    #
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         # indici delle img da attaccare ogni batch
    #         # può succedere che l'ultima batch sia più piccola rispetto alle altre e che il numero di attacchi da fare sia superiore
    #         # perciò setto queste condizione nel caso in cui ci sia questo caso
    #         # es : ultima batch: 3  --- attacchi per batch : 4 --> ERRORE ---> attacchi per batch : 4/2
    #         if len(inputs) < attack_per_batch and len(inputs) > 1:
    #             attack_per_batch = len(inputs) // 2
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #
    #         if len(inputs) > attack_per_batch:
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #         # se ultima batch : 1 ---> attacca l'unica img che hai
    #         if len(inputs) == 1:
    #             attack_per_batch = 1
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #
    #         for i in idx:
    #             adv = attack(inputs[i], targets[[i]])  # genera attacco
    #             inputs[i] = adv  # sostituisci img pulita con quella attaccata
    #             adv_num += 1  # aggiorna numero di attacchi generati
    #
    #         optimizer.zero_grad()
    #
    #         outputs = net(inputs)
    #
    #         loss = loss_func(outputs, targets)
    #
    #         running_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #
        total += targets.size(0)
        correct += torch.argmax(outputs, -1).eq(targets).cpu().sum().item()

    return losses, correct, total


def adv_validation(net, loader, loss_func, attack, type_of_attack,
                   attack_per_batch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    adv_num = 0
    total = 0
    adv_missclass = 0
    device = next(net.parameters()).device

    if type_of_attack == 'black':
        print('The attack is:', type_of_attack)
        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

                inputs, targets = inputs.to(device), targets.to(device)

                # indici delle img da attaccare ogni batch
                if len(inputs) < attack_per_batch and len(inputs) > 1:
                    attack_per_batch = len(inputs) // 2
                    idx = random.sample(range(len(inputs)), attack_per_batch)
                if len(inputs) > attack_per_batch:
                    idx = random.sample(range(len(inputs)), attack_per_batch)
                if len(inputs) == 1:
                    attack_per_batch = 1
                    idx = random.sample(range(len(inputs)), attack_per_batch)
                for i in idx:
                    adv = attack(inputs[i], targets[[i]])
                    inputs[i] = adv
                    adv_num += 1

                outputs = net(inputs)
                loss = loss_func(outputs, targets)

                test_loss += loss.item()

                # accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
                adv_missclass += (predicted[idx] != targets[idx]).sum().item()

            print('Accuratezza sul validation-set di img: {} %'.format(
                100 * (correct / total)))
            logger.info(f'Validation accuracy: {100 * (correct / total)}')
            print(
                'Rateo di successo dell attacco sul validation-set di img: {} %'.format(
                    100 * (adv_missclass / adv_num)))
            logger.info(
                f'Success rate on validation set: {100 * (adv_missclass / adv_num)}')

    else:
        print('The attack is:', type_of_attack)
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

            inputs, targets = inputs.to(device), targets.to(device)

            # indici delle img da attaccare ogni batch
            if len(inputs) < attack_per_batch and len(inputs) > 1:
                attack_per_batch = len(inputs) // 2
                idx = random.sample(range(len(inputs)), attack_per_batch)
            if len(inputs) > attack_per_batch:
                idx = random.sample(range(len(inputs)), attack_per_batch)
            if len(inputs) == 1:
                attack_per_batch = 1
                idx = random.sample(range(len(inputs)), attack_per_batch)
            adv_batch = attack(inputs, targets)

            for i in idx:
                inputs.data[i] = adv_batch.data[i]
                adv_num += 1

            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
            adv_missclass += (predicted[idx] != targets[idx]).sum().item()

        if total > 0:
            print('Accuratezza sul validation-set di img: {} %'.format(
                100 * (correct / total)))
            logger.info(f'Validation accuracy: {100 * (correct / total)}')
            print(
                'Rateo di successo dell attacco sul validation-set di img: {} %'.format(
                    100 * (adv_missclass / adv_num)))
            logger.info(
                f'Success rate on validation set: {100 * (adv_missclass / adv_num)}')

    return correct, adv_missclass, total


def adv_testing(net, loader, attack):
    net.eval()

    correct = 0
    total = 0

    attacked_images = 0
    adv_missclass = 0

    device = next(net.parameters()).device

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)

        pred = net(inputs)
        pred = torch.argmax(pred, -1)

        mask = pred == targets

        attacked_images += mask.sum().item()
        total += targets.size(0)

        inputs = inputs[mask]
        targets = targets[mask]

        if len(inputs) == 0 or inputs.shape[0] == 0:
            continue

        adv = attack(inputs, targets)

        predicted = torch.argmax(net(adv), 1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        adv_missclass += (predicted != targets).sum().item()

    correctly_attacked = adv_missclass / attacked_images

    if total > 0:
        logger.info(f'Images {total}, attacked images {attacked_images}')
        logger.info(f'Correctly attacked images: {correctly_attacked}')

    return total, attacked_images, correctly_attacked

    # if type_of_attack == 'black':
    #     print('The attack is:', type_of_attack)
    #     with torch.no_grad():
    #
    #         for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
    #
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             # indici delle img da attaccare ogni batch
    #             if len(inputs) < attack_per_batch and len(inputs) > 1:
    #                 attack_per_batch = len(inputs) // 2
    #                 idx = random.sample(range(len(inputs)), attack_per_batch)
    #             if len(inputs) > attack_per_batch:
    #                 idx = random.sample(range(len(inputs)), attack_per_batch)
    #             if len(inputs) == 1:
    #                 attack_per_batch = 1
    #                 idx = random.sample(range(len(inputs)), attack_per_batch)
    #             for i in idx:
    #                 adv = attack(inputs[i], targets[[i]])
    #                 inputs[i] = adv
    #                 adv_num += 1
    #
    #             outputs = net(inputs)
    #
    #             # accuracy
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets.data).cpu().sum().item()
    #             adv_missclass += (predicted[idx] != targets[idx]).sum().item()
    #
    #         print('Accuratezza sul test-set di img: {} %'.format(
    #             100 * (correct / total)))
    #         logger.info(f'Test accuracy: {100 * (correct / total)}')
    #         print(
    #             'Rateo di successo dell attacco sul test-set di img: {} %'.format(
    #                 100 * (adv_missclass / adv_num)))
    #         logger.info(
    #             f'Success rate on test set: {100 * (adv_missclass / adv_num)}')
    #
    # else:
    #     print('The attack is:', type_of_attack)
    #     for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
    #
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         # indici delle img da attaccare ogni batch
    #         if len(inputs) < attack_per_batch and len(inputs) > 1:
    #             attack_per_batch = len(inputs) // 2
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #         if len(inputs) > attack_per_batch:
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #         if len(inputs) == 1:
    #             attack_per_batch = 1
    #             idx = random.sample(range(len(inputs)), attack_per_batch)
    #         adv_batch = attack(inputs, targets)
    #         ''
    #         for i in idx:
    #             inputs.data[i] = adv_batch.data[i]
    #             adv_num += 1
    #
    #         outputs = net(inputs)
    #
    #         # accuracy
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets.data).cpu().sum().item()
    #         adv_missclass += (predicted[idx] != targets[idx]).sum().item()
    #
    #     print('Accuratezza sul test-set di img: {} %'.format(
    #         100 * (correct / total)))
    #     logger.info(f'Test accuracy: {100 * (correct / total)}')
    #     print('Rateo di successo dell attacco sul test-set di img: {} %'.format(
    #         100 * (adv_missclass / adv_num)))
    #     logger.info(
    #         f'Success rate on test set: {100 * (adv_missclass / adv_num)}')
