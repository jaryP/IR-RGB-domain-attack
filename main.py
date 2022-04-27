import argparse
import os

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from eval import calculate_iou_on_dataset
from utils.osu import OSU
from utils.plot import plot_boxes


def collate_fn(batch):
    return tuple(zip(*batch))


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True,
                    help="the name of the dataset", choices=['osu'])
parser.add_argument("--dataset_path", type=str, required=True,
                    help="the path of the dataset")
parser.add_argument("--dataset_domain", type=str, choices=['ir', 'rgb'],
                    default='rgb',
                    help='the domain of the dataset (Infrared or RGB)')

parser.add_argument("--test_split", type=float, required=False, default=None,
                    help="the dimension of the test slplit (used if the dataset does not have one)", )

parser.add_argument("--model_name", type=str, required=True,
                    help="the name of the model",
                    choices=['fasterrcnn_resnet50_fpn'])

parser.add_argument("--batch_size", type=int, default=4,
                    help="size of each image batch")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--seed", type=int, default=0,
                    help="the seed used during the experiments")

parser.add_argument("--save_path", type=str, default=None,
                    help="where to save the model")
parser.add_argument("--checkpoint_name", type=str, default=None,
                    help="the name used to save the model")
parser.add_argument('--save', '-s', action='store_true')
parser.add_argument('--load', '-l', action='store_true')

parser.add_argument("--device", type=str, default='cpu',
                    help="the devioce where to train the model")

# parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
# parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
# parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
# parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
# parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
# parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
# parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
# parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
# parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
# parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)
torch.random.manual_seed(opt.seed)

device = 'cpu' if not torch.cuda.is_available() or opt.device == 'cpu' else f'cuda:{opt.device}'

if opt.save_path is not None:
    os.makedirs(opt.save_path, exist_ok=True)

train_set = None
test_set = None

if opt.dataset == 'osu':
    train_set = OSU(opt.dataset_path, True, transforms=ToTensor())
    if opt.dataset_domain == 'ir':
        train_set.use_ir()
    else:
        train_set.use_rgb()
else:
    assert False

if test_set is None and opt.test_split is not None:
    train_size = int((1 - opt.test_split) * len(train_set))
    test_size = len(train_set) - train_size
    train_set, test_set = torch.utils.data.random_split(train_set,
                                                        [train_size,
                                                         test_size])

    print(len(test_set))

if opt.model_name == 'fasterrcnn_resnet50_fpn':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2,
                                                                 pretrained=False)
else:
    assert False

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

if test_set is not None:
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    print(len(test_loader))
else:
    test_loader = None

save_path = os.path.expanduser(opt.save_path) \
    if opt.save_path is not None else None

if save_path is not None and \
        os.path.exists(os.path.join(save_path, 'model.pt')) and opt.load:
    model.load_state_dict(torch.load(os.path.join(save_path,
                                                  opt.checkpoint_name),
                                     map_location=device))
else:
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())

    for epoch in range(opt.epochs):
        for batch_i, (imgs, targets) in enumerate(train_loader):

            imgs = [im.to(device) for im in imgs]

            targets = [{k: v.to(device) for k, v in t.items()} for t in
                       targets]

            optimizer.zero_grad()

            loss = model(imgs, targets)
            loss = sum(loss.values())

            loss.backward()
            optimizer.step()

            if (batch_i + 0) % 100 == 0:

                # print(calculate_iou_on_dataset(model, train_loader))
                plot_boxes(imgs, targets)

                if test_loader is not None:
                    print('Evaluating')
                    print(calculate_iou_on_dataset(model, test_loader))

                    # drawn_boxes = draw_bounding_boxes(img, boxes, colors="red")
                    # show(drawn_boxes)

                model.train()

                # model.eval()
                #
                # l = []
                # for t in targets:
                #     l.append({'boxes': t['boxes'],
                #               'labels': t['labels']})
                #
                # # l = {'boxes': [t['boxes'] for t in targets],
                # #      'labels': [t['labels'] for t in targets]}
                # gt = convert_box_to_mask(imgs, l, is_gt=True)
                #
                # loss = model(imgs, targets)
                # pred = convert_box_to_mask(imgs, loss)
                #
                # score = custom_iou(gt, pred)
                # print(score.mean())
                # # print(loss[0], pred[0].shape)
                # model.train()

    if save_path is not None and opt.save:
        torch.save(model.state_dict(),
                   os.path.join(os.path.join(save_path, opt.checkpoint_name), 'model.pt'))
