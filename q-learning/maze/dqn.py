import random
import torch
from torch import nn
from collections import namedtuple


class Model(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        layer_sizes = [in_features] + hidden
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(layer_sizes[-1], out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    import numpy as np
    model = Model(in_features=2, hidden=[12, 12], out_features=4)

    z = torch.tensor([1, 2], dtype=torch.float)
    print(model(z))
    z = torch.tensor([4, 5], dtype=torch.float)
    print(model(z))
    z = torch.tensor([0, 0], dtype=torch.float)
    print(model(z))


