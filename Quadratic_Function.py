import torch
import numpy as np


#######################################
#   Quadratic potential function      #
#######################################
# try Sigma = |  1   0  |
#             |  0   1  |
def Quadratic_Function_torch(z, dim):

    mu = 3 * torch.ones(dim)
    invSigma = 4.0 * torch.eye(dim)
    u = torch.mm(invSigma, (z-mu).t())
    Quadratic = torch.diag(0.5 * torch.mm(z-mu, u))

    return Quadratic