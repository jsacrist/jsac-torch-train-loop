# -*- coding: utf-8 -*-
# Imports from standard libraries
import datetime
from dataclasses import dataclass
from typing import Callable, Dict

# Imports from 3rd party libraries
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import writer as tb_writer

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
    data_loader: DataLoader,
    *,
    num_epochs: int = 10,
    log_freq: int = 100,
    #
    writer: tb_writer.SummaryWriter | None = None,
    validation_loader: DataLoader | None = None,
    eval_metrics: Dict[str, torch.nn.modules.loss._Loss] | None = None,
    #
    od_wait: int | None = None,
    checkpoint_dir: str = ".",
    keep_checkpoints: bool = False,
    #
    progress: bool | str = False,
    progress_level: int = 2,
    #
    feat_transform: Callable | None = None,
    label_transform: Callable | None = None,
    model_call_func: Callable | None = None,
    device: str = "cpu",
    verbose: bool = True,
):
    """General-purpose train-loop for PyTorch models, with some extra conveniences.

    This function aims at reducing the boilerplate code needed to train a
    `PyTorch`_ model, while providing a few convenient features during train
    time:

    - Integration with `TensorBoard`_ (via PyTorch's `SummaryWriter`_) for
      plotting:

        - **Training loss** (requires argument :attr:`writer`)
        - **Validation loss** (requires arguments :attr:`writer` and
          :attr:`validation_loader`)
        - Additional optional metrics (requires arguments :attr:`writer`
          and :attr:`eval_metrics`)

    - Progress bar(s) (via `tqdm`_) for Jupyter Notebooks or CLI environments.
    - `Early stopping`_ overfitting detection (requires arguments :attr:`od_wait` and
      :attr:`validation_loader`)

    Args:
        model (torch.nn.Module): A PyTorch model.
        criterion (torch.nn.modules.loss._Loss): Objective function used to
            train :attr:`model`.
        optimizer (Optimizer): `Optimizer`_ object that provides an
            optimization algorithm (e.g. `SGD`_ or `Adam`_).
        data_loader (DataLoader):
            `DataLoader`_ object used for updating parameters in the
            :attr:`model` (training).
        num_epochs (int, optional): Number of times to pass through all batches
            from :attr:`data_loader` during training. Defaults to 10.
        log_freq (int, optional): Number of consequtive batches (iterations)
            from :attr:`data_loader` to use before sending plot-data to
            :attr:`writer`. Defaults to 100.
        writer (tb_writer.SummaryWriter | None, optional):
            A `SummaryWriter`_ object to wich plotting data is sent every
            :attr:`log_freq` batches (iterations) . Defaults to None.
        validation_loader (DataLoader | None, optional):
            `DataLoader`_ object used for out-of-sample validation (not
            training). Defaults to None.
        eval_metrics (Dict[str, torch.nn.modules.loss._Loss] | None, optional):
            A dictionary containing loss-function objects to be used as
            evaluation metrics on the training set (:attr:`data_loader`).
            Keys in this dictionary should be strings by which the plotted
            metrics will be named. Defaults to None.
        od_wait (int | None, optional): Number of consequtive batches (iterations)
            from :attr:`data_loader` to wait for an improvement in the
            validation loss before stopping training to avoid overfitting.
            If this parameter is not passed, no overfitting-detection mechanism
            is engaged. Defaults to None.
        checkpoint_dir (str, optional): Directory in which to save checkpoint
            weights and biases used for `Early stopping`_ overfitting
            detection.  Defaults to ".".
        keep_checkpoints (bool, optional):  Whether or not to keep the
            checkpoints files (weight and biases) used for `Early stopping`_
            overfitting detection.  Defaults to False.
        progress (bool | str, optional):
            Defines which version of progress bar `tqdm`_ to use.
            If a string is provided, it must be one of ``notebook`` or ``cli``.
            If the boolean `True` is provided, the ``notebook`` version of tqdm
            will be used.  Defaults to False.
        progress_level (int, optional): Number of progress-bars to display

            - 1: Only show Epoch progress bar.
            - 2: Show Epoch and Batch progress bars.
            - 3: Show Epoch, Batch, and Validation set progress bars.

            Defaults to 2.
        feat_transform (Callable | None, optional): Transformation function to
            be used on the features of each batch (iteration) coming from
            :attr:`data_loader` (and from :attr:`validation_loader` if
            provided). Defaults to None.
        label_transform (Callable | None, optional): Transformation function to
            be used on the labels of each batch (iteration) coming from
            :attr:`data_loader` (and from :attr:`validation_loader` if
            provided). Defaults to None.
        model_call_func (Callable | None, optional): Custom function to be used
            to call the model.  This is useful in some cases where calling the
            model requires more parameters than just the feature array, or
            calling the model returns extra information (for example RNNs,
            which require and return hidden states).  If provided, the function
            signature should only accept `**kwargs` and return a dictionary
            with at least one key called `"outputs"`.  e.g.:

                >>> def model_caller(**kwargs):
                >>>     return {"outputs": kwargs["model"](kwargs["features"])}

            Defaults to None.
        device (str, optional): Device to which both the model and the
            data-batches will be sent before computing gradients.
            Defaults to "cpu".
        verbose (bool, optional): Determines whether or not to print out the
            progress status. Defaults to True.

    .. _PyTorch: https://github.com/pytorch/pytorch
    .. _TensorBoard: https://github.com/tensorflow/tensorboard
    .. _SummaryWriter: https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter#torch.utils.tensorboard.writer.SummaryWriter
    .. _tqdm: https://github.com/tqdm/tqdm
    .. _Early stopping: https://en.wikipedia.org/wiki/Early_stopping
    .. _DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    .. _Optimizer: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
    .. _SGD: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    .. _Adam: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
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
    epoch_losses = []
    idx_last_log = -1
    od_candidate = None

    # Main loop Start (epochs)
    bar_epoch = (
        tqdm(total=num_epochs, desc="Epoch", leave=True, dynamic_ncols=True)
        if progress
        else None
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
            model_call_result = {}
            if model_call_func:
                model_call_result = model_call_func(**locals())
            else:
                model_call_result["outputs"] = model(features)
            loss = criterion(model_call_result["outputs"], labels)

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
                        eval_values[metric_name] += metric(
                            model_call_result["outputs"], labels
                        ).item()

                # Depending on "log frequency": Log the loss value.
                if (validation_loader is not None or writer is not None) and (
                    (idx_batch + 1) % log_freq == 0
                    or idx_batch + 1 == n_batches
                ):
                    # Compute validation loss on the whole validation set (if provided)
                    if validation_loader is not None:
                        _loss_validation = 0.0
                        _len_val = len(validation_loader)
                        bar_val = (
                            tqdm(
                                total=_len_val,
                                desc="Validation",
                                leave=False,
                                dynamic_ncols=True,
                            )
                            if progress and progress_level >= 3
                            else None
                        )
                        for (
                            feat_validation,
                            lbl_validation,
                        ) in validation_loader:
                            # Validation Pre-process features and labels
                            feat_validation = feat_validation.to(device)
                            lbl_validation = lbl_validation.to(device)
                            if feat_transform:
                                feat_validation = feat_transform(
                                    feat_validation
                                )
                            if label_transform:
                                lbl_validation = label_transform(lbl_validation)

                            # Validation Forward pass
                            val_cal_result = {}
                            if model_call_func:
                                val_locals = locals()
                                val_locals["features"] = feat_validation
                                val_cal_result = model_call_func(**val_locals)
                            else:
                                val_cal_result["outputs"] = model(
                                    feat_validation
                                )
                            _loss_validation += criterion(
                                val_cal_result["outputs"], lbl_validation
                            ).item()
                            if bar_val is not None:
                                bar_val.update()
                        if bar_val is not None:
                            bar_val.close()
                        # Compute average validation loss on the whole validation set
                        loss_values["validation"] = _loss_validation / _len_val

                        #
                        if od_wait is not None:
                            if od_candidate is None:
                                # Save first checkpoint
                                od_candidate = ODCandidate(
                                    idx=idx_step,
                                    loss=loss_values["validation"],
                                    hash=h.hash_state_dict(model.state_dict()),
                                )
                                torch.save(
                                    model.state_dict(),
                                    h.get_model_path(
                                        checkpoint_dir, od_candidate
                                    ),
                                )
                            elif od_wait >= idx_step - od_candidate.idx:
                                if (
                                    loss_values["validation"]
                                    <= od_candidate.loss
                                ):
                                    # Remove previous checkpoint
                                    if not keep_checkpoints:
                                        h.remove_checkpoint(
                                            checkpoint_dir, od_candidate
                                        )
                                    # Save checkpoint
                                    od_candidate = ODCandidate(
                                        idx=idx_step,
                                        loss=loss_values["validation"],
                                        hash=h.hash_state_dict(
                                            model.state_dict()
                                        ),
                                    )
                                    torch.save(
                                        model.state_dict(),
                                        h.get_model_path(
                                            checkpoint_dir, od_candidate
                                        ),
                                    )
                            else:
                                print(
                                    f"Overfit Detection at step {idx_step} with loss={h.fmt_loss(loss_values['validation'])}\n"
                                    + f"Loading best model from step {od_candidate.idx} with loss={h.fmt_loss(od_candidate.loss)} ({hex(od_candidate.hash)})"
                                )
                                # Load model checkpoint
                                model.load_state_dict(
                                    torch.load(
                                        h.get_model_path(
                                            checkpoint_dir, od_candidate
                                        )
                                    )
                                )
                                # Remove previous checkpoint
                                if not keep_checkpoints:
                                    h.remove_checkpoint(
                                        checkpoint_dir, od_candidate
                                    )
                                break

                    # Figure out elapsed time, discard microseconds
                    delta = datetime.datetime.now() - start_time
                    delta -= datetime.timedelta(microseconds=delta.microseconds)

                    # Compute running AVERAGE loss and AVERAGE metrics
                    loss_values["train"] /= idx_step - idx_last_log
                    eval_values = h.div_dict(
                        eval_values, (idx_step - idx_last_log)
                    )

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
                writer.add_scalars(
                    "loss", {"train_epoch": epoch_losses[-1]}, idx_step
                )
                writer.flush()
            if bar_batch is not None:
                bar_batch.close()
            if bar_epoch is not None:
                bar_epoch.set_postfix_str(
                    f"Loss: {h.fmt_loss(epoch_losses[-1])}"
                )
                bar_epoch.update()
            continue

        # Only executed if the inner loop issues a "break statement"
        break
    # End of epoch loop
    if bar_epoch is not None:
        bar_epoch.close()
    if writer is not None:
        writer.close()
    # Remove previous checkpoint
    if od_wait is not None and not keep_checkpoints:
        h.remove_checkpoint(checkpoint_dir, od_candidate)
