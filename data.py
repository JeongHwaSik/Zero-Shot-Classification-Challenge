import os
import torch
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.datasets.utils import check_integrity, verify_str_arg



class ImageNetDataset(ImageFolder):
    """
    To load ImageNet dataset, there should be a ImageNet dataset in the root directory
    Plus, this class assumes that images are already splited into train & val folders
    e.g. root/ImageNet/train
    e.g. root/ImageNet/val
    """

    def __init__(self, root, split="train", transform=None, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(os.path.join(root, "imageNet")) # e.g. './imageNet' to '/home/username/imageNet'
        self.split = verify_str_arg(split, "split", ("train", "val"))

        # self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0] # dict(wnid: (classes, )) for 1000 classes

        super().__init__(self.split_folder, transform, **kwargs) # get images from './imageNet/train'
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

def load_meta_file(root, file:Optional[str]=None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = 'meta.bin'
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file, weights_only=True)
    else:
        msg = (
            "The meta file {} is not present in the root directory or is corrupted. "
            "This file is automatically created by the ImageNet dataset."
        )
        raise RuntimeError(msg.format(file, root))

def imagenet_classnames(root: str) -> List[str]:
    """
    Assume that you downloaded human-readable ImageNet class names and ImageNet datasets
    e.g. roor/imageNet
    e.g. root/imagenet_idx2cls.txt
    Obtained from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    """
    transform = transforms.Compose([
        transforms.CenterCrop(224, ),
        transforms.ToTensor()
    ])
    train_ds = ImageFolder('imageNet/train', transform=transform)
    class_names = train_ds.classes
    with open(os.path.join(root, 'imagenet_idx2cls.txt'), mode='r') as file:
        while True:
            content = file.readline()
            if not content:
                break
            idx = content.strip().split(" ")[0]
            cls = content.strip().split(" ")[2]
            class_names = list(map(lambda x: x.replace(idx, cls), class_names))
    return class_names


class SUN397Dataset(Dataset):
    """
    To load SUN397 dataset, there should be a SUN397 dataset folder
    e.g. root/SUN397
    """

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self._data_dir = Path(root) / "SUN397"

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files]

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image_file, label = self._image_files[index], self._labels[index]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def k_sample_per_image(dataset: Dataset, k: int=16, save=None) -> Dataset:

    # Get indices per classs
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(tqdm(dataset)):
        class_indices[label].append(idx)

    # Sample k images per class
    sampled_indices = [] # all image indices to be used for 16-shot
    for label, indices in class_indices.items():
        sampled_indices.extend(random.sample(indices, min(len(indices), k)))

    # Create a subset dataset with the sampled indices
    subset_ds = Subset(dataset, sampled_indices)
    if save:
        torch.save(subset_ds, os.path.join(save, f'{dataset.__class__.__name__}_{k}shot_sample.pt')) # save it for later use

    return subset_ds


def create_dataset(ft_dataset, root, transform, k_shot, batch_size, blurred=False):
    """
    Dataset for linear probing should be in the right path
    e.g. root/ImageNet
    e.g. root/SUN397
    """
    if blurred:
        blur = [transforms.GaussianBlur(kernel_size=(5, 9)) for _ in range(50)]
        transform = transforms.Compose(blur + transform.transforms)
    else:
        transform=transform

    if ft_dataset == "ImageNet":
        train_ds = ImageNet(root=os.path.join(root, 'imageNet'), split='train', transform=transform)
        class_names = imagenet_classnames(root)
    elif ft_dataset == "SUN397":
        train_ds = SUN397Dataset(root=root, transform=transform)
        class_names = train_ds.classes
    else:
        raise ValueError(f"Dataset {ft_dataset} not supported")

    # Few shot or Full shot
    if type(k_shot) == int:
        print(f"{ft_dataset} {k_shot}-shot fine-tuning")
        train_ds = k_sample_per_image(dataset=train_ds, k=k_shot)
    elif k_shot == 'full':
        print(f"{ft_dataset} {k_shot}-shot fine-tuning")
    else:
        raise ValueError(f"k-shot should be integer(2/4/8/16) or 'full'")

    # Dataloader
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=2)

    return train_loader, class_names


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.CenterCrop(224,),
        transforms.ToTensor()
    ])

    imagenet_ds = ImageNetDataset(root='./', split="train", transform=transform)
    print("ImageNet dataset testing...")
    for data in imagenet_ds:
        # data will have tuple (image, label)
        print(f"One image size: {data[0].shape}, label: {data[1]}")  # image with shape (3, 224, 224), label
        break


    # # Can also use torchvision.datasets.ImageNet()
    # ds = ImageNet(root='./imageNet', split='train', transform=transform)
    # print(len(ds.classes))
    # print(ds.class_to_idx)

    print("SUN397 dataset testing...")
    sun_ds = SUN397Dataset(root='./', transform=transform)
    for data in imagenet_ds:
        # data will have tuple (image, label)
        print(f"One image size: {data[0].shape}, label: {data[1]}")  # image with shape (3, 224, 224), label
        break