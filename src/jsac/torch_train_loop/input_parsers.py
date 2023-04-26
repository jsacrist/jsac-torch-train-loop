# -*- coding: utf-8 -*-
# Imports from standard libraries
# ...

# Imports from 3rd party libraries
# ...

# Imports from this project
# ...


__all__ = [  # External-facing members exported by this file
    #
    "parse_nonzero_positive_int",
    "parse_od_wait",
    "parse_transform_func",
    "parse_progress_level",
    "parse_progress",
    #
    "init_loss_value_dict",
    "init_eval_values_dict",
]


def parse_nonzero_positive_int(num: int):
    assert isinstance(num, int)
    assert num >= 1
    return num


def parse_od_wait(od_wait, validation_loader):
    if od_wait is None:
        return None
    assert validation_loader is not None
    return parse_nonzero_positive_int(od_wait)


def parse_transform_func(func):
    if func is None:
        return None
    assert callable(func)
    return func


def parse_progress_level(progress_level: int):
    progress_level = parse_nonzero_positive_int(progress_level)
    assert progress_level <= 3
    return progress_level


def parse_progress(progress: bool | str):
    if not progress:
        return False
    if progress is True:
        return "notebook"
    assert isinstance(progress, str)
    _supported = ["notebook", "cli"]
    assert (
        progress in _supported
    ), f"unsuported {progress}.  Provide one of {_supported}"
    return progress


def init_loss_value_dict(validation_loader):
    loss_values = {"train": 0.0}
    if validation_loader is not None:
        loss_values["validation"] = 0.0
    return loss_values


def init_eval_values_dict(eval_metrics, writer):
    eval_values = {}
    if eval_metrics is not None:
        assert writer is not None, "'eval_metrics' Requires 'writer'"
        eval_values = {
            metric_name: 0.0 for (metric_name, _) in eval_metrics.items()
        }
    return eval_values
