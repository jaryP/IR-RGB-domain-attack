import os
import sys
from typing import Optional, Callable, Tuple, List, cast, Any
from torchvision.datasets.folder import default_loader, \
    has_file_allowed_extension


def find_classes(dir):

    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if
                   os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def load_nir_dataset(directory):
    directory = os.path.expanduser(directory)
    _, class_to_idx = find_classes(directory)
    # image_path_list = sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])

    extensions = ('.tiff',)

    def is_valid_file(x: str) -> bool:
        return has_file_allowed_extension(x, cast(Tuple[str, ...],
                                                  extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    ir_instances = []
    rgb_instances = []
    instances = []

    available_classes = set()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)

        if not os.path.isdir(target_dir):
            continue

        for root, _, fnames in sorted(
                os.walk(target_dir, followlinks=True)):
            names = set(n.split('_')[0] for n in fnames)

            for fname in sorted(names):
                path1 = os.path.join(root, f'{fname}_rgb.tiff')
                path2 = os.path.join(root, f'{fname}_nir.tiff')

                if not (is_valid_file(path1) and is_valid_file(path2)):
                    continue

                item = path1, path2, class_index
                instances.append(item)

                ir_instances.append((path2, class_index))
                rgb_instances.append((path1, class_index))

                if target_class not in available_classes:
                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes

    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, ir_instances, rgb_instances


class PathDataset:
    def __init__(self,
                 paths: List[Tuple[str, int]],
                 is_ir: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        self.paths = paths
        self.transform = transform
        self.target_transform = target_transform
        self.is_ir = is_ir

        self.loader = default_loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.paths[index]

        sample = self.loader(path)
        if self.is_ir:
            sample = sample.convert('L').convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
