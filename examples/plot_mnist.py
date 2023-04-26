# -*- coding: utf-8 -*-
"""
MNIST example
====================
This example shows the usage of ``jsac.torch_train_loop.train()`` on the famous
MNIST dataset.
"""

# %%

# Imports
import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# sphinx_gallery_start_ignore
import os
import sys
import inspect

CURDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))
CODEDIR = os.path.realpath(os.path.join(CURDIR, "../"))
sys.path.insert(0, CODEDIR)
# sphinx_gallery_end_ignore
from jsac.torch_train_loop.core import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

# Hyperparameters
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 500
NUM_CLASSES = 10

NUM_EPOCHS = 5
BATCH_SIZE = 200
LEARNING_RATE = 0.02
LOG_FREQ = 100
ODWAIT = 400

# %%

# Writer
writer_datetime = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
writer = SummaryWriter(f"/tmp/runs/mnist/{writer_datetime}")

# %%

# Data objects
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
validation_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=1_000,
    shuffle=False,
)


# %%


# Neural Network definition
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size, device=device)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = out.squeeze(-2)
        return out


# %%

# Create an instance of the model
torch.manual_seed(204)
model = NeuralNet(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# %%

# Now for the good part, training
train(
    model,
    criterion=criterion,
    optimizer=optimizer,
    data_loader=train_loader,
    #
    num_epochs=NUM_EPOCHS,
    log_freq=LOG_FREQ,
    writer=writer,
    validation_loader=validation_loader,
    eval_metrics={
        "xent": torch.nn.CrossEntropyLoss(),
    },
    #
    od_wait=ODWAIT,
    feat_transform=lambda x: x.reshape(-1, 1, 28 * 28),
    label_transform=lambda x: torch.zeros(x.shape[0], 10).scatter(
        1, x.unsqueeze(1), 1.0
    ),
    device=device,
    verbose=False,
)
# While the model is training, you can run tensorboard to look at the plots in
# real-time: `tensorboard --bind_all --logdir=/tmp/runs/mnist/`

# %%

# We can also plot the metrics directly on a notebook as well, the lowest
# value of `loss_validation` coincides with the reported by the Early Stopping
# mechanism in the previous cell
traces = ["loss_train", "loss_validation", "loss_train_epoch"]
df = pd.DataFrame()
fig, ax = plt.subplots()
for trace in traces:
    reader = SummaryReader(
        writer.log_dir + "/" + trace, pivot=True, event_types={"scalars"}
    )
    trace_df = reader.scalars.set_index("step")
    trace_df.rename(columns={"loss": trace}, inplace=True)
    trace_df.plot(
        ax=ax,
        ylim=[0.04, 0.05],
    )
fig.show()


# %%

# Due to theEarly Stopping mechanism, the model with the best performance on the
# validation set was loaded:
model.eval()
_loss_validation = 0

feat_transform = lambda x: x.reshape(-1, 1, 28 * 28)
label_transform = lambda x: torch.zeros(x.shape[0], 10).scatter(
    1, x.unsqueeze(1), 1.0
)

for feat_validation, lbl_validation in validation_loader:
    feat_validation = feat_validation.to(device)
    lbl_validation = lbl_validation.to(device)

    feat_validation = feat_validation.reshape(-1, 1, 28 * 28)
    lbl_validation = torch.zeros(lbl_validation.shape[0], 10).scatter(
        1, lbl_validation.unsqueeze(1), 1.0
    )

    outputs_validation = model(feat_validation)
    _loss_validation += criterion(outputs_validation, lbl_validation).item()
print(_loss_validation / len(validation_loader))
