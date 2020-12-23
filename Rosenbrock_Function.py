import torch


##############################################################################
# Rosenbrock Function (reference: https://www.sfu.ca/~ssurjano/rosen.html)   #
##############################################################################
def Rosenbrock_Function_torch(z, dimension):

    z = torch.chunk(z, chunks=dimension, dim=1)

    Rosenbrock = 0
    for d in range(dimension-1):
        Rosenbrock = Rosenbrock + 10 * (z[d+1] - z[d] ** 2) ** 2 + (z[d] - 1) ** 2

    return 0.06 * Rosenbrock



