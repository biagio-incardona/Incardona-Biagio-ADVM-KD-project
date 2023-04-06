from mlp import MLPNet, CIFAR10_loaders
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import json

class SNet(torch.nn.Module):

    def __init__(self, dims):#=0.03, th=2.0, num_epochs=50):
        super(SNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        for d in range(len(dims) - 1):
            self.layers += [torch.nn.Linear(dims[d], dims[d + 1])]#.cuda()]

    def _set_activations(self, activations):
        self.activation = []
        for i in range(len(self.layers)-1):
            self.activation.append(activations[0])
        self.activation.append(activations[1])

    def compile(self, activations, optimizer, loss, num_epochs, lr):
        self._set_activations(activations)
        self.optimizer = optimizer
        self.loss = loss
        self.num_epochs = num_epochs

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation[i](h)
        return h

    def predict(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation[i](h)
        _,out = torch.max(h, dim=1)
        return out

    def train_model(self, orig_net, train_loader, val_loader):
        self.train()
        history_train = {}
        history_val = {}
        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                target = orig_net.forward(data)
                self.optimizer.zero_grad()
                output = self(data)
                loss_ = self.loss(output, target)
                epoch_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
            avg_loss = epoch_loss / len(train_loader)
            history_train[epoch+1] = self.accuracy(train_loader)
            history_val[epoch+1] = self.accuracy(val_loader)
            print(f"Epoch {epoch+1}: Train acc {history_train[epoch+1]:.9f}, Val acc {history_val[epoch+1]:.9f}%, Avg Train Loss {avg_loss:.4f}")

        return history_train, history_val

    def accuracy(self, loader):
        cumul = 0
        size = 0
        self.eval()
        for x, y in loader:
            size += len(x)

            cumul += float(self.\
                           predict(x)\
                            .eq(y)\
                            .float()\
                            .sum())
        return cumul/size
        
    def evaluate(self, orig_model, loader):
        # Set model to evaluation mode
        self.eval()
        cumul = 0
        size = 0
        for X,_ in loader:
        # Compute predictions on test data
            size += len(X)
            orig_out = orig_model.forward(X)
            output = self(X)
            # Compute loss on test data using mean squared error
            loss = torch.nn.functional.mse_loss(output, orig_out)
            cumul += float(loss.item())
        # Return loss value
        return cumul/size

def plot_accuracies(train_hist, val_hist):
    val_acc =val_hist.values()
    train_acc = train_hist.values()

    plt.plot(train_acc, '-x', label="train")
    plt.plot(val_acc, '-o', label="validation")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.title(f'Accuracy vs. No. of epochs')
    plt.show()
        #plt.savefig(f'{filename}_{key}_layers.png')
        #plt.clf()

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, val_loader, test_loader = CIFAR10_loaders(val_ratio=0.1)
    layers = [3072, 2048, 3000, 10]
    net = MLPNet(layers)

    net.compile(
                activations=[torch.nn.ReLU(), torch.nn.Softmax(dim=1)],
                optimizer=torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.5),
                loss=torch.nn.CrossEntropyLoss(),
                num_epochs = 30,
                lr=0.03
    )
    history_train, history_val= net.train_model(train_loader, val_loader)

    shrink_loss = torch.nn.MSELoss()
    layer_shrink = [3072, 100, 10]
    snet = SNet(layer_shrink)
    snet.compile(
        activations = [torch.nn.ReLU(), torch.nn.ReLU()],
        optimizer=torch.optim.Adam(snet.parameters()),
        loss=torch.nn.MSELoss(),
        num_epochs=20,
        lr=0.03
    )
    train, val = snet.train_model(net, train_loader, val_loader)
    plot_accuracies(train, val)