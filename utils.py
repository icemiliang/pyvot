# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 13th 2019

import numpy as np
from PIL import Image
import warnings
import matplotlib.pyplot as plt


color_blue = [0.12, 0.56, 1]
color_light_blue = [0.5, 0.855, 1]
color_dark_blue = [0.05, 0.28, 0.63]

color_red = [0.8, 0.22, 0]
color_light_red = [1.0, 0.54, 0.5]

color_light_grey = [0.7, 0.7, 0.7]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", ( w ,h ), buf.tostring())


def random_sample(num, dim, sampling='unisquare'):
    """ randomly sample the area with dirac measures

    """
    data = None

    if num * dim > 1e8:
        warnings.warn("Sampling the area will take too much memory.")
    if sampling == 'unisquare':
        data = np.random.random((num, dim)) * 2 - 1
    elif sampling == 'unicircle':
        r = np.random.uniform(low=0, high=1, size=num)  # radius
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)  # angle
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        data = np.concatenate((x[:, None], y[:, None]), axis=1)
    elif sampling == 'gaussian':
        mean = [0, 0]
        cov = [[.1, 0], [0, .1]]
        data = np.random.multivariate_normal(mean, cov, num).clip(-0.99, 0.99)

    label = -np.ones(num).astype(int)
    return data, label


def plot_map(data, idx, color_map='viridis'):
    color = plt.get_cmap(color_map)
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.title('Area-preserving mapping')
    plt.scatter(data[:, 0], data[:, 1], s=1, marker='o', color=color(idx))
    return fig
