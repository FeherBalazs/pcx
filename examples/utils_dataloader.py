import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    root_path: str,
    train_subset_n: int = None,
    test_subset_n: int = None,
    target_class: int = None
):
    """
    Returns train and test dataloaders for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('fashionmnist', 'cifar10', or 'imagenet').
        batch_size (int): Batch size for the dataloaders.
        root_path (str): Root directory where the dataset is stored.
        train_subset_n (int, optional): Number of samples to use from the training set.
        test_subset_n (int, optional): Number of samples to use from the test set.
        target_class (int, optional): If specified, filters the dataset to this class.
    """
    # Define dataset-specific transforms
    if dataset_name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_root = root_path + "fashion-mnist/"
        train_dataset = torchvision.datasets.FashionMNIST(
            root=dataset_root,
            transform=transform,
            download=True,
            train=True,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=dataset_root,
            transform=transform,
            download=False,
            train=False,
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_root = root_path + "cifar10/"
        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            transform=transform,
            download=True,
            train=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            transform=transform,
            download=False,
            train=False,
        )
    elif dataset_name == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root=root_path + "train",
            transform=transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=root_path + "val",
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # If target_class is specified, filter the dataset to only include that category
    if target_class is not None:
        if dataset_name in ["fashionmnist", "cifar10"]:
            target_indices = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
            train_dataset = Subset(train_dataset, target_indices)
            target_indices = (test_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
            test_dataset = Subset(test_dataset, target_indices)
        elif dataset_name == "imagenet":
            train_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == target_class]
            test_indices = [i for i, (_, label) in enumerate(test_dataset.samples) if label == target_class]
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)

    # Optionally restrict the datasets further
    if train_subset_n is not None:
        all_idx = list(range(len(train_dataset)))
        train_dataset = Subset(train_dataset, all_idx[:train_subset_n])
    if test_subset_n is not None:
        all_idx = list(range(len(test_dataset)))
        test_dataset = Subset(test_dataset, all_idx[:test_subset_n])

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader