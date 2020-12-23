import torch
import numpy as np


#######################################
#   Quadratic potential function      #
#######################################
# try Sigma = |  1   0    |
#             |  0   0.25 |
def Quadratic_Function2_torch(z, dim):

    mu = 3 * torch.ones(dim)
    invSigma = torch.eye(dim)
    invSigma[1][1] = torch.tensor(4.0)
    u = torch.mm(invSigma, (z-mu).t())
    Quadratic = torch.diag(0.5 * torch.mm(z-mu, u))

    return Quadratic
