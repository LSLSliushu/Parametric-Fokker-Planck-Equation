import torch
import torch.nn as nn
import torch.nn.functional as F


# Fully connected Neural Network with ReLU activation function
class ReLUNN(nn.Module):
    def __init__(self, network_length, hidden_dimension, input_dimension, output_dimension):
        super(ReLUNN, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length-2)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension)])

    def initialization(self):
        for l in self.linears:
            l.weight.data.uniform_(-0.1, 0.1)
            l.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        for l in self.linears[:-1]:
            x = l(x)
            x = F.relu(x)

        ll = self.linears[-1]
        x = ll(x)

        return x


# Fully connected Neural Network with Tanh activation functions
class TanhNN(nn.Module):
    def __init__(self, network_length, hidden_dimension, input_dimension, output_dimension):
        super(TanhNN, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(1, network_length-2)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension)])

    def initialization(self):
        for l in self.linears:
            l.weight.data.uniform_(-0.01, 0.01)
            l.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):

        for l in self.linears[:-1]:
            x = l(x)
            x = torch.tanh(x)

        ll = self.linears[-1]
        x = ll(x)

        return x
