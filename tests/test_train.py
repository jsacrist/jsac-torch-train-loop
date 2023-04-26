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
CODEDIR = os.path.realpath(os.path.join(CURDIR, "../src/"))
sys.path.insert(0, CODEDIR)
from jsac.torch_train_loop.core import train


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


@pytest.fixture
def my_feat_preproc():
    def func(x):
        return x.reshape(-1, 1, 28 * 28)

    return func


@pytest.fixture
def my_label_preproc():
    def func(x):
        return torch.zeros(x.shape[0], 10).scatter(1, x.unsqueeze(1), 1.0)

    return func


def test_passes_minimal_params(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    train(
        model=my_model,
        criterion=my_criterion,
        optimizer=my_optimizer,
        data_loader=my_train_loader,
        #
        num_epochs=1,
        feat_transform=my_feat_preproc,
        label_transform=my_label_preproc,
    )


def test_invalid_model(
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AttributeError):
        train(
            model=None,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_criterion(
    my_model,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(TypeError):
        train(
            model=my_model,
            criterion=None,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_optimizer(
    my_model,
    my_criterion,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AttributeError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=None,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_progress_level_low(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            progress_level=0,
        )


def test_invalid_progress_level_high(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            progress_level=4,
        )


def test_invalid_progress_level_float(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            progress_level=1.618,
        )


def test_invalid_log_freq_low(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            log_freq=0,
        )


def test_invalid_log_freq_lower(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            log_freq=-1,
        )


def test_invalid_log_freq_float(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            log_freq=3.14156,
        )


def test_invalid_num_epochs_low(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=0,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_num_epochs_lower(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=-1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_num_epochs_float(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=3.14156,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
        )


def test_invalid_eval_metrics_no_writer(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            eval_metrics={
                "mse": torch.nn.MSELoss(),
                "xent": torch.nn.CrossEntropyLoss(),
            },
        )


def test_invalid_od_wait_no_validation_loader(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            od_wait=10,
        )


def test_invalid_od_wait_low(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
    my_test_loader,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            od_wait=0,
            validation_loader=my_test_loader,
        )


def test_invalid_od_wait_float(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
    my_test_loader,
):
    with pytest.raises(AssertionError):
        train(
            model=my_model,
            criterion=my_criterion,
            optimizer=my_optimizer,
            data_loader=my_train_loader,
            #
            num_epochs=1,
            feat_transform=my_feat_preproc,
            label_transform=my_label_preproc,
            #
            od_wait=3.14159,
            validation_loader=my_test_loader,
        )


def test_passes_od_wait(
    my_model,
    my_criterion,
    my_optimizer,
    my_train_loader,
    my_feat_preproc,
    my_label_preproc,
    my_test_loader,
):
    train(
        model=my_model,
        criterion=my_criterion,
        optimizer=my_optimizer,
        data_loader=my_train_loader,
        #
        num_epochs=1,
        feat_transform=my_feat_preproc,
        label_transform=my_label_preproc,
        #
        od_wait=2,
        validation_loader=my_test_loader,
    )
