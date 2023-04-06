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
import time

def CIFAR10_loaders(val_ratio, train_batch_size=128, test_batch_size=128):
    transform = Compose(
                        [ToTensor(),
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])
    
    train_data = CIFAR10('./data/', train=True,
              download=True,
              transform=transform)
    
    test_data = CIFAR10('./data/', train=False,
              download=True,
              transform=transform)
    
    val_size = round(val_ratio * len(train_data))
    train_size = len(train_data) - val_size

    train_ds, val_ds = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size, shuffle=True)
    
    val_loader = DataLoader(
        val_ds,
        batch_size = val_size,
        shuffle = False)

    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MLPNet(torch.nn.Module):

    def __init__(self, dims):#=0.03, th=2.0, num_epochs=50):
        super(MLPNet, self).__init__()
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

    def train_model(self, train_loader, val_loader):
        start_time = time.time()
        history_train = {}
        history_val = {}
        history_time = {}
        self.train()
        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self(data)
                loss_ = self.loss(output, target)
                epoch_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
            avg_loss = epoch_loss / len(train_loader)
            history_train[epoch+1] = self.accuracy(train_loader)
            history_val[epoch+1] = self.accuracy(val_loader)
            print(f"Epoch {epoch+1}: Train Accuracy {history_train[epoch+1]:.2f}%, Val Accuracy {history_val[epoch+1]:.2f}%, Avg Train Loss {avg_loss:.4f}")
            end_time = time.time()
            time_elapsed = end_time - start_time
            history_time[epoch+1] = time_elapsed
        return history_train, history_val, history_time

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

def plot_accuracies(train_hist, val_hist, filename):
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
    layers_list = [
   #     [3072, 100, 3072, 10],
   #     [3072, 3072, 100, 10],
   #     [3072, 600, 500, 400, 10],
        [3072, 50, 40, 30, 20, 15, 10]#,
   #     [3072, 100, 80, 60, 40, 20, 10],
   #     [3072, 26, 24, 22, 20, 18, 16, 14, 12, 11, 10]
    #    [3072, 400, 10],
     #   [3072, 1500, 100, 50, 10],
      #  [3072, 10,10],
       # [3072, 1024,10, 10],
        #[3072, 6000, 50,10]
    ]

    layers_list = [
        [3072, 10],
        [3072, 20],
        [3072, 30],
        [3072, 50],
        [3072, 100],
        [3072, 200],
        [3072, 300],
        [3072, 500],
        [3072, 800],
        [3072, 1000],
        [3072, 1200],
        [3072, 1500],
        [3072, 1800],
        [3072, 2000],
        [3072, 2300],
        [3072, 2500],
        [3072, 3000]
    ]

    for layers in range(len(layers_list)):
        net = MLPNet(layers_list[layers])
        dim = len(layers_list[layers])
        net.compile(
            activations=[torch.nn.ReLU()] * dim,
            optimizer=torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.5),
            loss=torch.nn.CrossEntropyLoss(),
            num_epochs = 1,
            lr=0.03
        )
        history_train, history_val, history_time= net.train_model(train_loader, val_loader)

        to_dump = {
            'layers' : layers_list[layers][1],
            'time' : history_time[1]
            #'learning rates' : 0.005,
            #'epochs' : 30,
            #'train accuracy' : history_train,
            #'validation accuracy' : history_val
        }

        with open('results_mlp_time.json', 'a') as outfile:
            json.dump(to_dump, outfile)