# jsac's `torch_train_loop`

`torch_train_loop` is a general-purpose train-loop for [PyTorch][PyTorch] models, with some convenient features built-in:
  - Integration with [TensorBoard][TensorBoard] (via PyTorch's [SummaryWriter][SummaryWriter]) for
    plotting:
    - Training loss
    - Validation loss
    - Additional optional metrics
  - Progress bar(s) (via [tqdm][tqdm]) for Jupyter Notebooks or CLI environments.
  - [Early stopping][EarlyStopping] overfitting detection.


[PyTorch]: https://github.com/pytorch/pytorch
[TensorBoard]: https://github.com/tensorflow/tensorboard
[SummaryWriter]: https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter#torch.utils.tensorboard.writer.SummaryWriter
[tqdm]: https://github.com/tqdm/tqdm
[EarlyStopping]: https://en.wikipedia.org/wiki/Early_stopping
