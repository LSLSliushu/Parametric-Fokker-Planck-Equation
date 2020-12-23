import torch
import numpy as np


###############################################################
#   Quadratic potential function (Non-diagonal, Non-isotropic #
###############################################################
# try Sigma = diag( Sigma_A, I, Sigma_B, I, Sigma_C ), \mu = (mu_A, 0, mu_B, 0, mu_C)
# Sigma_A^-1 = | 2.5  1.5 |  Sigma_A^-1 = | 1.0  0.0 |  Sigma_C =  | 4.0   0.0 |
#              | 1.5  2.5 |               | 0.0  4.0 |             | 0.0   4.0 |
# mu_A = (1, 1)  mu_B = (1, 2) mu_C = (2, 3)
def Quadratic_Function_high_dim_torch(z, dim):

    mu = 0.0 * torch.ones(dim)
    invSigma = torch.eye(dim)

    # Sigma_A^{-1}
    invSigma[0][0] = torch.tensor(2.5)
    invSigma[0][1] = torch.tensor(1.5)
    invSigma[1][1] = torch.tensor(2.5)
    invSigma[1][0] = torch.tensor(1.5)
    # mu_A
    mu[0] = torch.tensor(1.0)
    mu[1] = torch.tensor(1.0)

    # Sigma_B^{-1}
    invSigma[4][4] = torch.tensor(1.)
    invSigma[5][5] = torch.tensor(4.)
    # mu_B
    mu[4] = torch.tensor(1.0)
    mu[5] = torch.tensor(2.0)

    # Sigma_C^{-1}
    invSigma[8][8] = torch.tensor(4.0)
    invSigma[9][9] = torch.tensor(4.0)
    # mu_C
    mu[8] = torch.tensor(2.)
    mu[9] = torch.tensor(3.0)

    u = torch.mm(invSigma, (z-mu).t())
    Quadratic = torch.diag(0.5 * torch.mm(z-mu, u))

    return Quadratic