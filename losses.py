import numpy as np
import torch
from torch import nn


def inner_product(v1, v2, n):

    S = .0
    for k in range(n):
        S = S + torch.dot(v1[k], v2[k])
    mean = S/n

    return mean


class Relative_Entropy(nn.Module):

    def __init__(self, potential, dimension):
        super().__init__()

        self.potential = potential
        self.dimension = dimension

    def forward(self, zk, log_jacobians):

        sum_of_log_jacobians = sum(log_jacobians)

        return (-sum_of_log_jacobians + self.potential(zk, self.dimension)).mean()


# The losses (JKO_loss & Wass_loss) are proposed by us
# Here zk = T_theta(z), gradients = grad psi(T_theta_0(z)), z~p
class JKO_loss(nn.Module):

    def __init__(self, potential, stepsize, dimension):
        super().__init__()

        self.potential = potential
        self.stepsize = stepsize
        self.dimension = dimension

    def forward(self, zk, log_jacobians, gradients):

        sum_of_log_jacobians = sum(log_jacobians)
        freeenergy = (-sum_of_log_jacobians + self.potential(zk, self.dimension)).mean()

        W = inner_product(gradients, zk, zk.size()[0])

        wasserstein_dist = W.mean()

        # One choice of "F":
        # C = 1 / self.stepsize
        # F = freeenergy + torch.mul(wasserstein_dist, C)

        # Another choice of "F":
        c = self.stepsize
        F = torch.mul(freeenergy, c) + wasserstein_dist

        return F


# this loss is the square of L2 loss of ||grad_psi(T_theta_0(x))-(T_theta(x)-T_theta_0(x))||_{L2}(p)
# Here s_original = T_theta_0(z), s_current = T_theta(z), gradients = grad psi(T_theta_0(z)), z~p
class Wass_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, s_original, s_current, gradients):

        z = s_current - s_original

        W = inner_product(gradients - z, gradients - z, s_original.size()[0])
        D = W.mean()

        return D


# The losses (JKO_loss_modified & Wass_loss_modified) are slight modifications  (c.f. Remark 4.9) of the original ones
# zk = T_theta(z), gradients = grad psi(T_theta_0(z)), z~p
class JKO_loss_modified(nn.Module):

    def __init__(self, potential, dimension, epsilon, h):
        super().__init__()

        self.potential = potential
        self.dimension = dimension
        self.epsilon = epsilon
        self.h = h

    def forward(self, zk, log_jacobians, gradients):

        ratio = self.epsilon / self.h

        sum_of_log_jacobians = sum(log_jacobians)
        entropy = (-sum_of_log_jacobians + self.potential(zk, self.dimension)).mean()

        W = inner_product(gradients, zk, zk.size()[0])
        wasserstein_dist_part = W.mean()

        F = entropy + ratio * wasserstein_dist_part

        return F


# this loss is the square of L2 loss of ||grad_psi(T_theta_0(x))-(T_theta(x)-T_theta_0(x))/delta||_{L2}(p)
# Here s_original = T_theta_0(z), s_current = T_theta(z), gradients=grad psi(T_theta_0(z)), z~p
class Wass_loss_modified(nn.Module):

    def __init__(self, epsilon):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, s_original, s_current, gradients):

        z = (s_current - s_original) / self.epsilon

        W = inner_product(gradients - z, gradients - z, s_original.size()[0])
        D = W.mean()

        return D
