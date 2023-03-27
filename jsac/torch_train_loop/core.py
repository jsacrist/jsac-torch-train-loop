# -*- coding: utf-8 -*-
# Imports from standard libraries
import datetime
from dataclasses import dataclass
from typing import Union, Dict, List, Callable

# Imports from 3rd party libraries
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import writer

# Imports from this project
from . import helpers as h
from . import input_parsers as p

__all__ = [  # External-facing members exported by this file
    "train",
]


@dataclass
class ODCandidate:
    idx: int
    loss: float
    hash: int


def train(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: Optimizer,
    data_loader: torch.utils.data.dataloader.DataLoader,
    *,
    validation_loader: torch.utils.data.dataloader.DataLoader | None = None,
    eval_metrics: List[Dict[str, torch.nn.modules.loss._Loss]] | None = None,
    writer: writer.SummaryWriter | None = None,
    log_freq: int = 100,
    od_wait: Union[int, None] = None,
    device: str = "cpu",
    num_epochs: int = 10,
    feat_transform: Union[Callable, None] = None,
    label_transform: Union[Callable, None] = None,
    progress: Union[bool, str, None] = None,
    progress_level: int = 2,
    verbose: bool = True,
):
    """_summary_

    Args:
        model (torch.nn.Module): A pytorch model.
        criterion (torch.nn.modules.loss._Loss): _description_
        optimizer (Optimizer): _description_
        data_loader (torch.utils.data.dataloader.DataLoader):
            _description_
        validation_loader (torch.utils.data.dataloader.DataLoader | None, optional):
            _description_. Defaults to None.
        eval_metrics (List[Dict[str, torch.nn.modules.loss._Loss]] | None, optional):
            _description_. Defaults to None.
        writer (writer.SummaryWriter | None, optional):
            _description_. Defaults to None.
        log_freq (int, optional): _description_. Defaults to 100.
        od_wait (Union[int, None], optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".
        num_epochs (int, optional): _description_. Defaults to 10.
        feat_transform (Union[Callable, None], optional):
            _description_. Defaults to None.
        label_transform (Union[Callable, None], optional):
            _description_. Defaults to None.
        progress (Union[bool, str, None], optional):
            _description_. Defaults to None.
        progress_level (int, optional): _description_. Defaults to 2.
        verbose (bool, optional): _description_. Defaults to True.
    """
    # Parse and validate parameters
    log_freq = p.parse_nonzero_positive_int(log_freq)
    od_wait = p.parse_od_wait(od_wait, validation_loader)
    num_epochs = p.parse_nonzero_positive_int(num_epochs)
    feat_transform = p.parse_transform_func(feat_transform)
    label_transform = p.parse_transform_func(label_transform)
    progress = p.parse_progress(progress)
    progress_level = p.parse_progress_level(progress_level)

    #
    if progress == "notebook":
        from tqdm.notebook import tqdm
    elif progress:  # "cli"
        from tqdm import tqdm

    # Init dictionaries
    loss_values = p.init_loss_value_dict(validation_loader)
    eval_values = p.init_eval_values_dict(eval_metrics, writer)

    # Initialize variables
    n_batches = len(data_loader)
    model.to(device=device)
    start_time = datetime.datetime.now()
    epoch_losses = list()
    idx_last_log = -1
    od_candidate = None

    # Main loop Start (epochs)
    bar_epoch = (
        tqdm(total=num_epochs, desc="Epoch", leave=True, dynamic_ncols=True) if progress else None
    )
    for idx_epoch in range(num_epochs):
        # Zero-out losses
        batch_loss = 0.0
        loss_values = h.zero_out_dict(loss_values)
        eval_values = h.zero_out_dict(eval_values)

        # Inner loop start (batches)
        bar_batch = (
            tqdm(total=n_batches, desc="Batch", leave=False, dynamic_ncols=True)
            if progress and progress_level >= 2
            else None
        )
        for idx_batch, (features, labels) in enumerate(data_loader):
            # Get current step number
            idx_step = idx_epoch * n_batches + idx_batch

            # Pre-process features and labels
            features = features.to(device)
            labels = labels.to(device)
            if feat_transform:
                features = feat_transform(features)
            if label_transform:
                labels = label_transform(labels)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # ZBS: Zero-out gradients, Backprop pass, Step to update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of loss values
            batch_loss += loss.item()
            loss_values["train"] += loss.item()

            with torch.no_grad():
                # On every step: Compute/accumulate all evaluation metrics on the training set
                if eval_metrics is not None:
                    for metric_name, metric in eval_metrics.items():
                        eval_values[metric_name] += metric(outputs, labels).item()

                # Depending on "log frequency": Log the loss value.
                # TODO: Refactor this section so as to decrease cyclomatic complexity
                if (idx_batch + 1) % log_freq == 0 or idx_batch + 1 == n_batches:
                    # Compute validation loss on the whole validation set (if provided)
                    if validation_loader is not None:
                        _loss_validation = 0.0
                        _len_val = len(validation_loader)
                        bar_val = (
                            tqdm(total=_len_val, desc="Validation", leave=False, dynamic_ncols=True)
                            if progress and progress_level >= 3
                            else None
                        )
                        for feat_validation, lbl_validation in validation_loader:
                            # Validation Pre-process features and labels
                            feat_validation = feat_validation.to(device)
                            lbl_validation = lbl_validation.to(device)
                            if feat_transform:
                                feat_validation = feat_transform(feat_validation)
                            if label_transform:
                                lbl_validation = label_transform(lbl_validation)

                            # Validation Forward pass
                            outputs_validation = model(feat_validation)
                            _loss_validation += criterion(outputs_validation, lbl_validation).item()
                            if bar_val is not None:
                                bar_val.update()
                        if bar_val is not None:
                            bar_val.close()
                        # Compute average validation loss on the whole validation set
                        loss_values["validation"] = _loss_validation / _len_val

                        #
                        if od_wait is not None:
                            if od_candidate is None:
                                # TODO: save to file
                                od_candidate = ODCandidate(
                                    idx=idx_step,
                                    loss=loss_values["validation"],
                                    hash=h.hash_state_dict(model.state_dict()),
                                )
                            elif od_wait >= idx_step - od_candidate.idx:
                                if loss_values["validation"] <= od_candidate.loss:
                                    # TODO: save to file
                                    od_candidate = ODCandidate(
                                        idx=idx_step,
                                        loss=loss_values["validation"],
                                        hash=h.hash_state_dict(model.state_dict()),
                                    )
                            else:
                                print(
                                    f"Overfit Detection at step {idx_step} with loss={h.fmt_loss(loss_values['validation'])} "
                                    + f"Best model at step {od_candidate.idx} with loss={h.fmt_loss(od_candidate.loss)} ({hex(od_candidate.hash)})"
                                )
                                # TODO: Load from file
                                break

                    # Figure out elapsed time, discard microseconds
                    delta = datetime.datetime.now() - start_time
                    delta -= datetime.timedelta(microseconds=delta.microseconds)

                    # Compute running AVERAGE loss and AVERAGE metrics
                    loss_values["train"] /= idx_step - idx_last_log
                    eval_values = h.div_dict(eval_values, (idx_step - idx_last_log))

                    # Log to tensorboard writer
                    if writer is not None:
                        writer.add_scalars("loss", loss_values, idx_step)
                        if eval_metrics is not None:
                            writer.add_scalars("metrics", eval_values, idx_step)
                        writer.flush()

                    # Log to stdout
                    if verbose:
                        h.verbose_log(
                            loss_train_avg=loss_values["train"],
                            delta=delta,
                            idx_batch=idx_batch,
                            idx_epoch=idx_epoch,
                            num_epochs=num_epochs,
                            n_total_batches=n_batches,
                        )

                    # Reset running loss and evaluation metrics
                    loss_values = h.zero_out_dict(loss_values)
                    eval_values = h.zero_out_dict(eval_values)
                    idx_last_log = idx_step
                #
            if bar_batch is not None:
                bar_batch.update()
            #

        else:
            # End of batch loop (inner loop)
            epoch_losses.append(batch_loss / n_batches)
            if writer is not None:
                writer.add_scalars("loss", {"train_epoch": epoch_losses[-1]}, idx_step)
                writer.flush()
            if bar_batch is not None:
                bar_batch.close()
            if bar_epoch is not None:
                bar_epoch.set_postfix_str(f"Loss: {h.fmt_loss(epoch_losses[-1])}")
                bar_epoch.update()
            continue

        # Only executed if the inner loop issues a "break statement"
        break
    # End of epoch loop
    if bar_epoch is not None:
        bar_epoch.close()
    if writer is not None:
        writer.close()
