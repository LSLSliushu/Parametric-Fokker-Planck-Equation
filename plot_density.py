import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kde
import scipy.stats as st


# color plot of given samples
def plot_density(samples, a, b, nbins, iteration, L, path):

    X_LIMS = (-L, L)
    Y_LIMS = (-L, L)

    fig = plt.figure(figsize=(L, L))
    ax = fig.add_subplot(111)

    points = samples.data.numpy()[:, [a, b]]

    # compute KDE
    x, y = points.T
    k = kde.gaussian_kde(points.T)

    xi, yi = np.mgrid[-L:L:nbins * 1j, -L:L:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # draw the density
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues)
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title('Density on {}-{} plane at {} iteration'.format(a, b, iteration))

    name = os.path.join(path, "Density at {}=iteration (on {}-{} plane) ".format(iteration, a, b))
    fig.savefig(name)
    plt.close()


# plot probability density with contours of given samples
def plot_contour_density(samples, a, b, nbins, iteration, L, save_path):

    fig = plt.figure(figsize=(L, L))
    ax = fig.add_subplot(111)

    np_samples = samples.detach().numpy()

    points = np_samples[:, [a, b]]

    x, y = points.T

    xx, yy = np.mgrid[-L:L:nbins*1j, -L:L:nbins*1j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[-L, L, -L, L])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('{}th component'.format(a))
    ax.set_ylabel('{}th component'.format(b))
    plt.title('Density estimation')

    name = os.path.join(save_path, "Probability density with contours at {}=iteration (on {}-{} plane) ".format(iteration, a, b))
    fig.savefig(name)
    plt.close()


# plot 3D probability density function of given samples
def plot_3d_density(samples, a, b, nbins, iteration, L, save_path):

    fig = plt.figure(figsize=(L, L))
    ax = fig.add_subplot(111)

    np_samples = samples.detach().numpy()

    points = np_samples[:, [a, b]]

    x, y = points.T

    xx, yy = np.mgrid[-L:L:nbins*1j, -L:L:nbins*1j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('{}th component'.format(a))
    ax.set_ylabel('{}th component'.format(b))
    ax.set_zlabel('PDF')
    ax.set_title('Density')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)

    name = os.path.join(save_path, " Density (3D) at {}=iteration (on {}-{} plane) ".format(iteration, a, b))
    fig.savefig(name)
    plt.close()
