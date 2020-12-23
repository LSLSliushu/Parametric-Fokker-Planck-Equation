from matplotlib import pyplot as plt
import os


# plot sample points on the [-L,L]x[-L,L] window of 2-dimensional a-b plane
def plot_sample(a, b, points, iteration, flow_length, L, path):

    X_LIMS = (-L, L)
    Y_LIMS = (-L, L)

    fig = plt.figure(figsize=(L, L))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, a], points[:, b], alpha=0.7, s=25)
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title(
        "Flow length: {}\n Samples at {}th time step, on the {}-{}-Plane"
        .format(flow_length, iteration, a, b)
    )

    name = os.path.join(path, "Sample at {}th time step on the {}-{} Plane.png".format(iteration, a, b))
    fig.savefig(name)
    plt.close()
