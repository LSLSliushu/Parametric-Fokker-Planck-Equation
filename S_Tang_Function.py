import numpy as np
import torch
import math


#########################################################################################
#  Styblinski -Tang function   (reference: https://www.sfu.ca/~ssurjano/stybtang.html)  #
#########################################################################################

def S_Tang_single_var(z):

    return z**4-16*z**2+5*z


def S_Tang_Function_torch(z, dimension):

    z = torch.chunk(z, chunks=dimension, dim=1)

    S_Tang = 0.0
    for d in range(dimension):
        S_Tang = S_Tang + S_Tang_single_var(z[d])

    return 0.06 * S_Tang
