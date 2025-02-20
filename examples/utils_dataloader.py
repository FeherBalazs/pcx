import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

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
    batch_size: int,
    train_subset_n: int = None,
    test_subset_n: int = None,
    target_class: int = None
):
    t = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=True,
    )
    
    # If target_class is specified, filter the dataset to only include that category.
    if target_class is not None:
        target_indices = (train_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        train_dataset = Subset(train_dataset, target_indices)

    # Optionally restrict the training dataset further.
    if train_subset_n is not None:
        all_idx = list(range(len(train_dataset)))
        train_dataset = Subset(train_dataset, all_idx[:train_subset_n])

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
        transform=t,
        download=True,
        train=False,
    )
    
    # If target_class is specified, filter to only that category.
    if target_class is not None:
        target_indices = (test_dataset.targets == target_class).nonzero(as_tuple=True)[0].tolist()
        test_dataset = Subset(test_dataset, target_indices)

    # Similarly, restrict the test dataset if required.
    if test_subset_n is not None:
        all_idx = list(range(len(test_dataset)))
        test_dataset = Subset(test_dataset, all_idx[:test_subset_n])

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader 