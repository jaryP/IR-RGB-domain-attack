import numpy as np
import torch


def convert_to_xyxy(bboxes):
    for bbox in bboxes:
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
    return bboxes


def iou(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))  # Will be zzero if both are 0

    iou = (intersection + smooth) / (
            union + smooth)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0,
                              10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def custom_iou(gt, predictions, smooth=1e-6):

    scores = []
    for g, p in zip(gt, predictions):
        intersection = (g & p).float().sum()
        union = (g | p).float().sum()

        iou = (intersection + smooth) / (union + smooth)

        # This is equal to comparing with thresholds
        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

        scores.append(iou)

    return torch.tensor(scores)  # Or thresholded.mean() if you are interested in average across the batch


def convert_box_to_mask(image, predictions, is_gt=False):
    masks = []

    for img_i, p in enumerate(predictions):
        mask = torch.zeros_like(image[img_i], dtype=torch.long)

        boxes = [b.cpu().detach().numpy() for b in p['boxes']]
        # boxes = np.round(convert_to_xyxy(boxes))
        boxes = np.round(boxes).astype(int)

        labels = [l.item() for l in p['labels']]
        scores = p.get('scores', None)

        for box, lab in zip(boxes, labels):
            mask[box[1]: box[3], box[0]: box[2]] = lab

        # if is_gt:
        #     pass
        #     # for box, lab, score in zip(boxes, labels):
        #     #     box = box.cpu().detach().numpy()
        #     #     box = np.round(box).astype(int)
        #     #
        #     #     mask[box[1]: box[3], box[0]: box[2]] = lab.item()
        # else:
        #     for box, lab, score in zip(boxes, labels, scores):
        #         # box = box.cpu().detach().numpy()
        #         # box = np.round(box).astype(int)
        #         mask[box[1]: box[3], box[0]: box[2]] = lab

        masks.append(mask)

    return masks


@torch.no_grad()
def calculate_iou_on_dataset(model, dataset):
    device = next(model.parameters()).device

    model.eval()

    tot = 0
    iou = 0

    for imgs, targets in dataset:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in
                             targets]

        l = []
        for t in targets:
            l.append({'boxes': t['boxes'],
                      'labels': t['labels']})

        gt = convert_box_to_mask(imgs, l, is_gt=True)

        pred = convert_box_to_mask(imgs, model(imgs, targets))

        score = sum(custom_iou(gt, pred))

        iou += score
        tot += len(imgs)

    return iou / tot