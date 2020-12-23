import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def safe_log(z):
    return torch.log(z + 1e-7)


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            flow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            flow_log_det_Jacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = []

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        zk = z

        return zk, log_jacobians


class flow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        weight_initialbd = 0.1
        scale_initialnd = 0.1
        bias_initialbd = 0.1

        self.weight.data.uniform_(-weight_initialbd, weight_initialbd)
        self.scale.data.uniform_(-scale_initialnd, scale_initialnd)
        self.bias.data.uniform_(-bias_initialbd, bias_initialbd)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)

        return z + self.scale * self.tanh(activation)


class flow_log_det_Jacobian(nn.Module):

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight  # please replace "self.tanh" by "F.tanh"
        det_grad = 1 + torch.mm(psi, self.scale.t())

        return safe_log(det_grad.abs())
