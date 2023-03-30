# -*- coding: utf-8 -*-
# Imports from standard libraries
import os
import re
import struct

# Imports from 3rd party libraries
import torch

# Imports from this project
# ...


__all__ = [  # External-facing members exported by this file
    "fmt_loss",
    "zero_out_dict",
    "div_dict",
    "verbose_log",
    #
    "hash_tensor",
    "hash_state_dict",
    #
    "get_model_path",
    "remove_checkpoint",
]


BITSIZE = 8 * struct.calcsize("P")


def fmt_loss(loss_val, sep="_", decimals=6):
    loss_re_str = r"(\d)(?=(\d{3})+(?!\d))"
    return re.sub(loss_re_str, rf"\1{sep}", f"{loss_val:.{decimals}f}")


def zero_out_dict(some_dict: dict):
    some_dict = {k: 0.0 for k, _ in some_dict.items()}
    return some_dict


def div_dict(some_dict: dict, denominator: float):
    some_dict = {k: v / denominator for k, v in some_dict.items()}
    return some_dict


def verbose_log(*, loss_train_avg, delta, idx_batch, idx_epoch, num_epochs, n_total_batches):
    log_str = (
        ""
        + f"{delta}; "
        + f"Epoch [{idx_epoch + 1}/{num_epochs}]; "
        + f"Batch [{idx_batch + 1}/{n_total_batches}]; "
        + f"Average Loss: {fmt_loss(loss_train_avg)}; "
    )
    print(log_str)


def hash_tensor(t):
    assert isinstance(t, torch.Tensor)
    if t.dim() == 0:
        return hash(t.tolist())
    if t.dim() == 1:
        return hash(tuple(t.tolist()))

    # if there are more dimensions, hash each one-dimensional slice individually
    hlist = list()
    for el in t:
        hlist.append(hash_tensor(el))
    return hash(tuple(hlist))


def hash_state_dict(sd):
    hlist = list()
    for k, v in sd.items():
        assert isinstance(k, str)
        hlist.append((hash(k), hash_tensor(v)))
    return hash(tuple(hlist)) & ((1 << BITSIZE) - 1)


def get_model_path(checkpoint_dir, candidate):
    return os.path.join(checkpoint_dir, f"checkpoint.{candidate.idx}.{hex(candidate.hash)}.pt")


def remove_checkpoint(checkpoint_dir, candidate):
    checkpoint_path = get_model_path(checkpoint_dir, candidate)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
