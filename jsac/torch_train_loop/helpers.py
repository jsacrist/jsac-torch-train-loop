# -*- coding: utf-8 -*-
# Imports from standard libraries
from typing import Union
import re

# Imports from 3rd party libraries
# ..

# Imports from this project
# ...


__all__ = [  # External-facing members exported by this file
    "fmt_loss",
    "zero_out_dict",
    "div_dict",
    "verbose_log",
    #
    "parse_progress_level",
    "parse_progress",
    #
    "init_loss_value_dict",
    "init_eval_values_dict",
]


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


def parse_progress_level(progress_level: int):
    assert isinstance(progress_level, int)
    assert progress_level >= 1
    assert progress_level <= 3
    return progress_level


def parse_progress(progress: Union[bool, str, None]):
    if progress is None:
        return False
    if progress is True:
        return "notebook"
    assert isinstance(progress, str)
    _supported = ["notebook", "cli"]
    assert progress in _supported, f"unsuported {progress}.  Provide one of {_supported}"
    return progress


def init_loss_value_dict(validation_loader):
    loss_values = {"train": 0.0}
    if validation_loader is not None:
        loss_values["validation"] = 0.0
    return loss_values


def init_eval_values_dict(eval_metrics, tb_writer):
    eval_values = dict()
    if eval_metrics is not None:
        assert tb_writer is not None, "'val_metrics' Requires 'tb_writer'"
        eval_values = {metric_name: 0.0 for (metric_name, _) in eval_metrics.items()}
    return eval_values
