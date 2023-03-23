# -*- coding: utf-8 -*-
# Imports from standard libraries
import datetime
from typing import Union, Dict, List
import re

# Imports from 3rd party libraries
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import writer

# Imports from this project
from . import helpers as h

__all__ = [  # External-facing members exported by this file
    "train",
]


def train(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: Optimizer,
    data_loader: torch.utils.data.dataloader.DataLoader,
    *,
    validation_loader: torch.utils.data.dataloader.DataLoader | None = None,
    eval_metrics: List[Dict[str, torch.nn.modules.loss._Loss]] | None = None,
    tb_writer: writer.SummaryWriter | None = None,
    log_freq: int = 100,
    device: str = "cpu",
    num_epochs: int = 10,
    feat_premodel_func: Union[callable, None] = None,
    label_premodel_func: Union[callable, None] = None,
    progress: Union[bool, str, None] = None,
    progress_level: int = 2,
    verbose: bool = True,
):
    # Parse and validate parameters
    progress = h.parse_progress(progress)
    progress_level = h.parse_progress_level(progress_level)

    #
    if progress == "notebook":
        from tqdm.notebook import tqdm
    else:  # "cli"
        from tqdm import tqdm

    # Init dictionaries
    loss_values = h.init_loss_value_dict(validation_loader)
    eval_values = h.init_eval_values_dict(eval_metrics, tb_writer)

    # Initialize variables
    n_batches = len(data_loader)
    model.to(device=device)
    start_time = datetime.datetime.now()
    epoch_losses = list()
    idx_last_log = -1

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
            # Pre-process features and labels
            if feat_premodel_func:
                features = feat_premodel_func(features)
            features = features.to(device)
            if label_premodel_func:
                labels = label_premodel_func(labels)
            labels = labels.to(device)

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
                # Compute evaluation metrics on the training set
                if eval_metrics is not None:
                    for metric_name, metric in eval_metrics.items():
                        eval_values[metric_name] += metric(outputs, labels).item()

                # Log the loss value based on the "log frenquency"
                if (idx_batch + 1) % log_freq == 0 or idx_batch + 1 == n_batches:
                    trace_idx = idx_epoch * n_batches + idx_batch

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
                            if feat_premodel_func:
                                feat_validation = feat_premodel_func(feat_validation)
                            feat_validation = feat_validation.to(device)
                            if label_premodel_func:
                                lbl_validation = label_premodel_func(lbl_validation)
                            lbl_validation = lbl_validation.to(device)

                            # Validation Forward pass
                            outputs_validation = model(feat_validation)
                            _loss_validation += criterion(outputs_validation, lbl_validation).item()
                            if bar_val is not None:
                                bar_val.update()
                        if bar_val is not None:
                            bar_val.close()
                        # Compute average validation loss on the whole validation set
                        loss_values["validation"] = _loss_validation / _len_val

                    # Figure out elapsed time, discard microseconds
                    delta = datetime.datetime.now() - start_time
                    delta -= datetime.timedelta(microseconds=delta.microseconds)

                    # Compute running AVERAGE loss and AVERAGE metrics
                    loss_values["train"] /= trace_idx - idx_last_log
                    eval_values = h.div_dict(eval_values, (trace_idx - idx_last_log))

                    # Log to tensorboard writer
                    if tb_writer is not None:
                        tb_writer.add_scalars("loss", loss_values, trace_idx)
                        if eval_metrics is not None:
                            tb_writer.add_scalars("metrics", eval_values, trace_idx)
                        tb_writer.flush()

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
                    idx_last_log = trace_idx
                #
            if bar_batch is not None:
                bar_batch.update()
            #

        # End of batch loop
        epoch_losses.append(batch_loss / n_batches)
        if tb_writer is not None:
            trace_idx = idx_epoch * n_batches + idx_batch
            tb_writer.add_scalars("loss", {"train_epoch": epoch_losses[-1]}, trace_idx)
            tb_writer.flush()
        if bar_batch is not None:
            bar_batch.close()
        if bar_epoch is not None:
            bar_epoch.set_postfix_str(f"Loss: {h.fmt_loss(epoch_losses[-1])}")
            bar_epoch.update()
    # End of epoch loop
    if bar_epoch is not None:
        bar_epoch.close()
    if tb_writer is not None:
        tb_writer.close()
