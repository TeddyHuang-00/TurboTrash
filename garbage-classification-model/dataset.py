from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms
from torchvision.datasets import ImageFolder


def get_datasets(root_dir: str):
    TRAIN = Path(root_dir + "_train")
    TEST = Path(root_dir + "_test")

    train_transforms = transforms.Compose(
        [
            transforms.ToImage(),  # Convert PIL image to torch.Tensor
            transforms.RandomResizedCrop(
                (256, 256), scale=(0.8, 1.2), antialias=True
            ),  # Resize to 256x256
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomEqualize(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToImage(),  # Convert PIL image to torch.Tensor
            transforms.Resize((256, 256), antialias=True),  # Resize to 256x256
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_set = ImageFolder(TRAIN, transform=train_transforms)
    test_set = ImageFolder(TEST, transform=test_transforms)

    return train_set, test_set
