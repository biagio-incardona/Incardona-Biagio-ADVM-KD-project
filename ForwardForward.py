import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import json
import time


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

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
        batch_size = train_batch_size,
        shuffle = False)

    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class FFNet(torch.nn.Module):

    def __init__(self, dims, lr, th, num_epochs):#=0.03, th=2.0, num_epochs=50):
        super().__init__()
        assert len(lr) == len(th) == len(num_epochs) == len(dims)-1, "length of lr, th and num epochs must be = len of dims - 1"
        self.unused_layers = []
        self.trained_layers = []
        for d in range(len(dims) - 1):
            self.unused_layers += [FFLayer(dims[d], dims[d + 1], lr=lr[d], th=th[d], num_epochs=num_epochs[d])]#.cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.trained_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, train_loader, val_loader, name):
        #h_pos, h_neg = x_pos, x_neg
        history_train = {}
        history_val = {}
        history_time = {}
        layers_to_train = len(self.unused_layers)
        data = train_loader
        for i in range(layers_to_train):
            curr_layer = self.unused_layers.pop(0)
            print('training layer', i, '...')
            self.trained_layers.append(curr_layer)
            start_time = time.time()
            data, train_layer_history, val_layer_history = curr_layer.train(data, val_loader, i, self, train_loader)
            end_time = time.time()
            time_elapsed = end_time - start_time
            #val_acc = self.accuracy(val_loader)
            #print(val_acc)
            history_train[i+1] = train_layer_history
            history_val[i+1] = val_layer_history
            history_time[i+1] = time_elapsed
        #plot_accuracies(history_train, history_val, f".\\Desktop\\tests\\net_{name}_with")

        return history_train, history_val, history_time

    def accuracy(self, loader):
        cumul = 0
        size = 0
        for x, y in loader:
            #x, y = next(iter(loader))
            size += len(x)
            cumul += self.predict(x).eq(y).float().sum()
        return cumul/size
    
class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None,
                 lr=0.03, th=2.0, num_epochs=50):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = th
        self.num_epochs = num_epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def training_step(self,x_pos,x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
        return loss

    def train(self, train, val, n_layer, net, tl):
        h_s = []
        layer_history_train = []  
        layer_history_val = []
        for i in tqdm(range(self.num_epochs)):
            for x, y in train:
                if n_layer == 0:
                    #x, y = x.cuda(), y.cuda()
                    x_pos = overlay_y_on_x(x, y)
                    rnd = torch.randperm(x.size(0))
                    x_neg = overlay_y_on_x(x, y[rnd])
                else:
                    x_pos=x
                    x_neg=y

                loss = self.training_step(x_pos, x_neg)
                # this backward just compute the derivative and hence
                # is not considered backpropagation.
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                if i == self.num_epochs-1:
                    h_s.append((self.forward(x_pos).detach(), self.forward(x_neg).detach()))
            train_epoch_accuracy = net.accuracy(tl)
            val_epoch_accuracy = net.accuracy(val)
            print(f"\n epoch {i} train accuracy: {train_epoch_accuracy}")
            print(f"\n epoch {i} validation accuracy: {val_epoch_accuracy}")
            print(val_epoch_accuracy.float().numpy())
            layer_history_val.append(float(val_epoch_accuracy.float().numpy()))
            layer_history_train.append(float(train_epoch_accuracy.float().numpy()))
            #history.append(net.accuracy())
        return h_s, layer_history_train, layer_history_val

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(32, 32, 3)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
def plot_accuracies(train_hist, val_hist, filename):
    for key in train_hist.keys():
        train_history = train_hist[key]
        val_history = val_hist[key]
        #accuracies = [x['val_acc'] for x in history]
        plt.plot(train_history, '-x', label="train")
        plt.plot(val_history, '-o', label="validation")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='upper left')
        plt.title(f'Accuracy vs. No. of epochs with {key} layers')
        plt.savefig(f'{filename}_{key}_layers.png')
        plt.clf()

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, val_loader, test_loader = CIFAR10_loaders(val_ratio=0.1)

#    nets = [
#        [3072, 3072, 3072, 10],
#        [3072, 2048,2048],
#        [3072, 500, 300],
#        [3072, 1000, 800],
#        [3072, 100, 500],
#        [3072, 500, 100, 10],
#        [3072, 3072, 10],
#        [3072, 2048, 1024],
#        [3072, 2000]
#    ]

    nets = [
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
    lrs = [#=0.03, th=2.0, num_epochs=50):
        [0.03, 0.1],
        [0.03, 0.05],
        [0.03, 0.08],
        [0.1, 0.1],
        [0.1,0.05,0.1],
        [0.1,0.01],
        [0.05,0.07],
        [0.06],

        [0.04, 0.02],
        [0.06, 0.08],
        [0.07, 0.03],
        [0.03, 0.05],
        [0.1,0.1],
        [0.04]
    ]
    ths = [
        [2, 1],
        [3, 5],
        [3, 8],
        [1, 1],
        [1,0.5,0.1],
        [4,0.1],
        [5,0.07],
        [6]
    ]

    epochs = [
        [1, 30],
        [1, 40],
        [70, 35],
        [40, 30],
        [50, 50, 30],
        [60, 60], #here
        [30, 35],
        [35]
    ]

    for i in range(len(nets)):
        torch.manual_seed(i)
        dim = len(nets[i]) - 1
        net = FFNet(nets[i], [0.1] * dim, [1] * dim, [1] * dim)
        history_train, history_val, history_time = net.train(train_loader, val_loader, f"n_{nets[i]}")
        print(history_train)
        to_dump = {
            'layers' : nets[i][1],
            #'learning rates' : lrs[i],
            #'thresholds' : ths[i],
            #'epochs' : epochs[i],
            #'train accuracy' : history_train,
            #'validation accuracy' : history_val,
            'time' : history_time[1]
        }
        with open('results_time_ff.json', 'a') as outfile:
            json.dump(to_dump, outfile)
        
    
    
    #for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #    visualize_sample(data, name)
    
    
    #print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    #x_te, y_te = next(iter(test_loader))
    #x_te, y_te = x_te, y_te#.cuda(), y_te.cuda()

    #print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())