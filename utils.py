import torch


# Generating samples from the reference probability measure p
# (One may also try other reference probability measures)
def random_normal_samples(n, dim):
    return torch.zeros(n, dim).normal_(mean=0, std=1)


# create list of m numbers [n, 2n, 3n,...,mn] for plotting loss curves
def create_nodes(m, n):
    nodes = torch.ones(m, 1)
    for i in range(1, m + 1):
        nodes[i-1] = i * n
    return nodes


# create nodes for plotting the graph of psi
def psi_nodes(interval_width, num, dim):

    NUM = (num+1) * (num+1)

    torchv = torch.ones(NUM, dim)

    stepsize = 2 * interval_width / num

    for k in range(1, num+2):
        for l in range(1, num+2):
            torchv[(k-1)*(num+1)+(l-1)][0] = (k-1)*stepsize-interval_width
            torchv[(k-1)*(num+1)+(l-1)][1] = (l-1)*stepsize-interval_width

    return torchv
