import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    """Dataset wrapper. Takes numpy arrays and stores them as torch tensors.
    Data is transformed to the device."""

    def __init__(self, x: np.ndarray, y: np.ndarray, device):
        # self.x = torch.from_numpy(x).float().to(device)
        # self.y = torch.from_numpy(y).float().to(device)

        self.x = x
        self.y = y
        self.len = self.x.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return self.len


def split_data(
    x_full: np.ndarray, y_full: np.ndarray, train_fraction: float, device: str
):
    """Generates three pytorch Datasets given features and labels for
    the full data as well as split into  train and test Datasets."""

    # define fraction of data used for training
    assert x_full.shape[0] == y_full.shape[0]

    n_samples = y_full.shape[0]

    n_train = int(train_fraction * n_samples)

    # get indices for training and test set
    ids = np.arange(n_samples)
    np.random.shuffle(ids)
    ids_train, ids_test = np.split(ids, [n_train])

    all_data_dist = Data(x_full, y_full, device)
    train_data_dist = Data(x_full[ids_train], y_full[ids_train], device)
    test_data = Data(x_full[ids_test], y_full[ids_test], device)

    return all_data_dist, train_data_dist, test_data


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    device,
):
    """
    Basic training loop for a pytorch model.

    Parameters:
    -----------
    model : pytorch model.
    train_loader : pytorch Dataloader containing the training data.
    optimizer: Optimizer for gradient descent.
    criterion: Loss function.

    Example usage:
    -----------

    model = (...) # a pytorch model
    criterion = (...) # a pytorch loss
    optimizer = (...) # a pytorch optimizer
    trainloader = (...) # a pytorch DataLoader containing the training data

    epochs = 10000
    log_interval = 1000
    for epoch in range(1, epochs + 1):
        loss_train = train(model, trainloader,optimizer, criterion)

        if epoch % log_interval == 0:
            print(f'Train Epoch: {epoch} Loss: {loss_train:.6f}')
        losses_train.append(loss_train)


    """
    # Set model to training mode
    model.train()
    epoch_loss = 0
    n_batches = len(train_loader)

    # Loop over each batch from the training set
    for data, target in train_loader:
        # Copy data to device
        data = data.to(device)
        target = target.to(device)

        # set optimizer to zero grad to remove previous gradients
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # get gradients
        loss.backward()

        # gradient descent
        optimizer.step()

        epoch_loss += loss.data.item()

    return epoch_loss / n_batches


class MLP(nn.Module):
    def __init__(self, n_units: list, activation=nn.ReLU()):
        """
        Simple multi-layer perceptron (MLP).

        Parameters:
        -----------
        n_units : List of integers specifying the dimensions of input and output and the hidden layers.
        activation: Activation function used for non-linearity.

        Example:
        -----------

        dim_hidden = 100
        dim_in = 2
        dim_out = 5

        # MLP with input dimension 2, output dimension 5, and 4 hidden layers of dimension 100
        model = MLP([dim_in,
                    dim_hidden,
                    dim_hidden,
                    dim_hidden,
                    dim_hidden,
                    dim_out],activation=nn.ReLU()).to(DEVICE)

        """
        super().__init__()

        # Get input and output dimensions
        dims_in = n_units[:-1]
        dims_out = n_units[1:]

        layers = []

        # Add linear layers (and activation function after all layers except the final one)
        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(torch.nn.Linear(dim_in, dim_out))

            if i < len(n_units) - 2:
                layers.append(activation)

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

    def count_parameters(self):
        """
        Counts the number of trainable parameters.

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
