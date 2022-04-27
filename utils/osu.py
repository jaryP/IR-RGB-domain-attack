import os
from itertools import chain

import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, extract_archive, \
    download_and_extract_archive
from torchvision.transforms import ToTensor

from utils.dataset_base import BaseObjectDetectionDataset
import xml.etree.ElementTree as ET



class OSU(BaseObjectDetectionDataset):
    urls = [f'http://vcipl-okstate.org/pbvs/bench/Data/03/{i}a.zip'
            for i in range(1, 7)]

    rbg_urls = [f'http://vcipl-okstate.org/pbvs/bench/Data/03/{i}b.zip'
                for i in range(1, 7)]

    def __init__(self, root_path, download, transforms=None, target_transforms=None):
        self.download_dataset = os.path.abspath(os.path.join(root_path, 'download'))
        self.extract_dataset = os.path.abspath(os.path.join(root_path, 'dataset'))

        self.paths = []
        self.rgb_paths = []
        self.boxes = []

        self._use_rgb = False

        super().__init__(root_path=root_path,
                         download=download,
                         transforms=transforms,
                         target_transforms=target_transforms)

    def use_rgb(self):
        self._use_rgb = True

    def use_ir(self):
        self._use_rgb = False

    def load_xml(self, path):
        result = {}

        tree = ET.parse(path)
        frames = tree.findall(".//frame")

        for frame in frames:
            number = int(frame.attrib['number'])

            objects = frame.findall(".//object")

            obj = []

            for o in objects:
                i = int(o.attrib['id'])
                xi = float(o[1].attrib['x'])
                yi = float(o[1].attrib['y'])
                xe = float(o[1].attrib['w']) + xi
                ye = float(o[1].attrib['h']) + yi

                d = {'id': i,
                     'xs': xi, 'ys': yi,
                     'xe': xe, 'ye': ye,
                     # 'orientation': orientation
                     }
                obj.append(d)

            result[number] = obj

        return result

    def load_dataset(self):

        # for url in chain(self.urls, self.rbg_urls):
        # for url in self.urls:

        for i in range(1, 7):

            # for url in self.urls:
            # filename = os.path.basename(url).split('.')[0]

            # folder_path = os.path.join(self.extract_dataset, filename)

            ir_path = os.path.join(self.extract_dataset, f'{i}a')
            rgb_path = os.path.join(self.extract_dataset, f'{i}b')

            xml_path = os.path.join(self.extract_dataset,
                                    f'Thermal_0{i}.cvml')

            d = self.load_xml(xml_path)

            # print(filename, xml_path)
            # tree = ET.parse(xml_path)
            # print(tree.findall(".//frame")[0].findall(".//object")[0][1].attrib['x'])
            ir_files = sorted(os.listdir(ir_path))
            rgb_files = sorted(os.listdir(rgb_path))

            for ir_filename, rgb_filename in zip(ir_files, rgb_files):

                ir_frame_id = int(ir_filename.split('_')[1].split('.')[0])
                rgb_frame_id = int(rgb_filename.split('_')[1].split('.')[0])

                bbx = d.get(ir_frame_id, d.get(rgb_frame_id, ()))

                # continue
                # file_path = os.path.join(folder_path, filename)
                # bbx = d.get(frame_id, ())

                if len(bbx) == 0:
                    continue
                else:
                    bbx = [(b['xs'], b['ys'], b['xe'], b['ye'])
                           for b in bbx]
                    bbx = [b for b in bbx if sum(b) > 0]

                    if len(bbx) == 0:
                        continue

                # s = file_path.split('/')
                # s[-2] = s[-2].replace('a', 'b')
                # rgb_path = '/'.join(s)
                #
                # ir_path = file_path
                # # rgb_path =
                #
                # if os.path.exists(rgb_path) and os.path.isfile(file_path):

                # if 'a' in file_path.split('/')[-2]:
                    # if len(bbx) > 0:
                    #     bbx = [(b['xs'], b['ys'], b['xe'], b['ye'])
                    #            for b in bbx]
                    #
                self.paths.append(os.path.join(ir_path, ir_filename))
                self.boxes.append(bbx)
                # else:
                self.rgb_paths.append(os.path.join(rgb_path, rgb_filename))

                # print(folder_path, filename)

        # for url in self.rbg_urls:
        #     filename = os.path.basename(url).split('.')[0]
        #     folder_path = os.path.join(self.extract_dataset, filename)
        #
        #     for filename in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, filename)
        #         self.rbg_urls.append(file_path)
        #
        #         print(folder_path, filename)

    def download(self) -> None:
        for url in chain(self.urls, self.rbg_urls):
            filename = os.path.basename(url)

            if not os.path.isfile(os.path.join(self.download_dataset, filename)):
                download_url(url, self.download_dataset, filename=filename)

                archive = os.path.join(self.download_dataset, filename)
                extract_archive(archive, self.extract_dataset, False)

        download_and_extract_archive('http://vcipl-okstate.org/pbvs/bench/Data/03/tracks-Leykin.zip', download_root=self.download_dataset, extract_root=self.extract_dataset)

    def check_existence(self) -> bool:
        return False

    def get_boxes(self, idx) -> np.ndarray:
        return self.boxes[idx]

    def get_image(self, idx):
        if self._use_rgb:
            img = Image.open(self.rgb_paths[idx]).convert("RGB")
        else:
            img = Image.open(self.paths[idx])

        return img

    def get_labels(self, idx, image, masks, boxes):
        return np.ones((len(boxes),))

    def is_crowd(self, idx, image, masks) -> bool:
        return False

    def __len__(self) -> int:
        return len(self.paths)


if __name__ == '__main__':
    from utils.plot import plot_boxes
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def collate_fn(batch):
        return tuple(zip(*batch))

    # dataset = OSU('../dataset/OSU', True, transforms=ToTensor())
    dataset = OSU('../dataset/OSU', True)
    dataset.use_rgb()

    img, targets = dataset[1000]
    img = np.asarray(img)

    print(img.shape)
    print(torch.round(targets['boxes']).type(torch.ByteTensor))
    print(targets['boxes'].type(torch.ByteTensor))

    # plot_boxes([img], [targets])

    fig, ax = plt.subplots()

    # Display the image

    print(type(img))
    # print(np.moveaxis(img, 0, -2).shape)
    ax.imshow(img)

    # Create a Rectangle patch

    for box in targets['boxes']:
        box = box.numpy()

        x, y = box[:2]
        xe, ye = box[2:]

        w = xe - x
        h = ye - y
        print(x, y, xe, ye, w, h)

        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

    # # print(len(dataset.rgb_paths), len(dataset.paths))
    # # input()
    #
    # import torchvision
    # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # import torchvision
    # from torchvision.models.detection import FasterRCNN
    # from torchvision.models.detection.rpn import AnchorGenerator
    #
    # # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    #     pretrained=True)
    #
    # # replace the classifier with a new one, that has
    # # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #
    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # # FasterRCNN needs to know the number of
    # # output channels in a backbone. For mobilenet_v2, it's 1280
    # # so we need to add it here
    # backbone.out_channels = 1280
    #
    # # let's make the RPN generate 5 x 3 anchors per spatial
    # # location, with 5 different sizes and 3 different aspect
    # # ratios. We have a Tuple[Tuple[int]] because each feature
    # # map could potentially have different sizes and
    # # aspect ratios
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # # let's define what are the feature maps that we will
    # # use to perform the region of interest cropping, as well as
    # # the size of the crop after rescaling.
    # # if your backbone returns a Tensor, featmap_names is expected to
    # # be [0]. More generally, the backbone should return an
    # # OrderedDict[Tensor], and in featmap_names you can choose which
    # # feature maps to use.
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    #
    # # put the pieces together inside a FasterRCNN model
    # model = FasterRCNN(backbone,
    #                    num_classes=1,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)
    #
    # # import transforms as T
    # #
    # #
    # # def get_transform(train):
    # #     transforms = []
    # #     transforms.append(T.ToTensor())
    # #     if train:
    # #         transforms.append(T.RandomHorizontalFlip(0.5))
    # #     return T.Compose(transforms)
    #
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    #     pretrained=True)
    #
    # print(len(dataset))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=collate_fn)
    #
    # optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    #
    # for epoch in range(10):
    #     for batch_i, (imgs, targets) in enumerate(data_loader):
    #
    #         optimizer.zero_grad()
    #
    #         loss = model(imgs, targets)
    #         loss = sum(loss.values())
    #         print(loss)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    # # For Training
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images, targets)  # Returns losses and detections
    # print(output)
    #
    # # For inference
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions)
    #
    # # device = torch.device(
    # #     'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # #
    # # # our dataset has two classes only - background and person
    # # num_classes = 2
    # # # use our dataset and defined transformations
    # # # dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # # # dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    # #
    # # # split the dataset in train and test set
    # # # indices = torch.randperm(len(dataset)).tolist()
    # # # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # # # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # #
    # # # define training and validation data loaders
    # # data_loader = torch.utils.data.DataLoader(
    # #     dataset, batch_size=2, shuffle=True, num_workers=4,
    # #     collate_fn=utils.collate_fn)
    # #
    # # # data_loader_test = torch.utils.data.DataLoader(
    # # #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
    # # #     collate_fn=utils.collate_fn)
    # #
    # # # get the model using our helper function
    # # model = get_model_instance_segmentation(num_classes)
    # #
    # # # move model to the right device
    # # model.to(device)
    # #
    # # # construct an optimizer
    # # params = [p for p in model.parameters() if p.requires_grad]
    # # optimizer = torch.optim.SGD(params, lr=0.005,
    # #                             momentum=0.9, weight_decay=0.0005)
    # # # and a learning rate scheduler
    # # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    # #                                                step_size=3,
    # #                                                gamma=0.1)
    # #
    # # # let's train it for 10 epochs
    # # num_epochs = 10
    # #
    # # for epoch in range(num_epochs):
    # #     # train for one epoch, printing every 10 iterations
    # #     train_one_epoch(model, optimizer, data_loader, device, epoch,
    # #                     print_freq=10)
    # #     # update the learning rate
    # #     lr_scheduler.step()
    # #     # evaluate on the test dataset
    # #     evaluate(model, data_loader_test, device=device)
    # #
    # # print("That's it!")