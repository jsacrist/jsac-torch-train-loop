# -*- coding: utf-8 -*-
# Imports from standard libraries
import inspect
import os
import sys

# Imports from 3rd party libraries
import pytest
import torch
import torchvision

# Imports from this project
CURDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))
CODEDIR = os.path.realpath(os.path.join(CURDIR, "../"))
sys.path.insert(0, CODEDIR)
from jsac.torch_train_loop import train


#
BATCH_SIZE = 10


class SmallerMNIST(torch.utils.data.Dataset):
    def __init__(self, size, root, train, transform, download):
        self.size = size
        self.full_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.full_dataset[idx]


@pytest.fixture
def my_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(start_dim=-2),
        torch.nn.Linear(28 * 28, 10),
    )


@pytest.fixture
def my_criterion():
    return torch.nn.MSELoss()


@pytest.fixture
def my_optimizer(my_model):
    return torch.optim.Adam(my_model.parameters(), lr=0.01)


@pytest.fixture
def my_train_dataset():
    return SmallerMNIST(
        size=231,
        root=f"{CURDIR}/.cached",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )


@pytest.fixture
def my_test_dataset():
    return SmallerMNIST(
        size=100,
        root=f"{CURDIR}/.cached",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )


@pytest.fixture
def my_train_loader(my_train_dataset):
    return torch.utils.data.DataLoader(
        dataset=my_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


@pytest.fixture
def my_test_loader(my_test_dataset):
    return torch.utils.data.DataLoader(
        dataset=my_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def test_passes_minimal_params(my_model, my_criterion, my_optimizer, my_train_loader):
    train(
        model=my_model,
        criterion=my_criterion,
        optimizer=my_optimizer,
        data_loader=my_train_loader,
        #
        feat_premodel_func=lambda x: x.reshape(-1, 1, 28 * 28),
        label_premodel_func=lambda x: torch.zeros(x.shape[0], 10).scatter(1, x.unsqueeze(1), 1.0),
        num_epochs=1,
    )


def test_invalid_model(my_criterion, my_optimizer, my_train_loader):
    with pytest.raises(AttributeError):
        train(
            model="something_else",
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            feat_premodel_func=lambda x: x.reshape(-1, 1, 28 * 28),
            label_premodel_func=lambda x: torch.zeros(x.shape[0], 10).scatter(
                1, x.unsqueeze(1), 1.0
            ),
            num_epochs=1,
        )


def test_invalid_criterion(my_model, my_optimizer, my_train_loader):
    with pytest.raises(Exception):
        train(
            model=my_model,
            criterion="something else",
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            feat_premodel_func=lambda x: x.reshape(-1, 1, 28 * 28),
            label_premodel_func=lambda x: torch.zeros(x.shape[0], 10).scatter(
                1, x.unsqueeze(1), 1.0
            ),
            num_epochs=1,
        )
