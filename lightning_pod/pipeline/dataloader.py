import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST as torch_dataset

"""
use this file to retrieve the desired dataset from torchtext,
torchaudio, torchvision, or lightning-bolts

import the torch or lightning-bolts dataset as torch_dataset, as shown in
the import statements above.

then, if needed, add kwargs (key word arguments) for the dataset call in get_data(), in the
call to _fetch_data.

An example is below, where transform=transforms.ToTensor() is the kwarg:

` dataset = _fetch_data(torch_dataset, datapath, transform=transforms.ToTensor())`
"""


def _check_datacache_exists(datapath):
    if not os.path.isdir(datapath):
        os.mkdir(datapath)
    return


def _download_data(torch_dataset, datapath, **kwargs):
    dataset = torch_dataset(
        datapath,
        download=True,
        **kwargs,
    )
    return dataset


def _fetch_data(torch_dataset, datapath, **kwargs):
    directory_is_empty = os.listdir(datapath)
    if directory_is_empty:
        dataset = torch_dataset(
            datapath,
            download=False,
            **kwargs,
        )
    else:
        dataset = _download_data(torch_dataset, datapath, **kwargs)
    return dataset


def _split_data(dataset, train_size: float):
    train_set_size = int(len(dataset) * train_size)
    test_set_size = len(dataset) - train_set_size
    train, test = random_split(dataset, lengths=[train_set_size, test_set_size])
    return train, test


def _make_data(
    dataset,
    return_loader: bool,
    split: bool,
    num_workers: int,
    train_size: float = 0.8,
):
    if return_loader:
        if split:
            train, test = _split_data(dataset, train_size)
            train = DataLoader(train, num_workers=num_workers)
            test = DataLoader(test, num_workers=num_workers)
            data = train, test
        else:
            data = DataLoader(dataset, num_workers=num_workers)
    else:
        if split:
            data = _split_data(dataset, train_size)
        else:
            data = dataset
    return data


def get_data(
    return_loader: bool = True,
    split: bool = True,
    train_size: float = 0.8,
    num_workers: int = 0,
):
    rootpath = os.getcwd()
    datapath = os.path.join(rootpath, "data", "cache")
    _check_datacache_exists(datapath)
    dataset = _fetch_data(torch_dataset, datapath, transform=transforms.ToTensor())
    dataset = _make_data(
        dataset,
        return_loader,
        split,
        num_workers,
        train_size=train_size,
    )
    return dataset
