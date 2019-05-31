# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 30th 2019

import numpy as np
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import torch


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
    w, h = buf.shape[0], buf.shape[1]
    return Image.frombytes("RGBA", (w, h), buf.tostring())


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
    # close previously opened figure
    # TODO return figure and close it outside the function seems not good
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.title('Area-preserving mapping')
    plt.scatter(data[:, 0], data[:, 1], s=1, marker='o', color=color(idx))
    return fig


def rigid_transform_3d_pytorch(p1, p2):
    center_p1 = torch.mean(p1, dim=0, keepdim=True)
    center_p2 = torch.mean(p2, dim=0, keepdim=True)

    pp1 = p1 - center_p1
    pp2 = p2 - center_p2

    H = torch.mm(pp1.t(), pp2)
    U, S, Vt = torch.svd(H)
    R = torch.mm(Vt.t(), U.t())

    # reflection
    if np.linalg.det(R.cpu().numpy()) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = torch.mm(Vt.t(), U.t())

    t = torch.mm(-R, center_p1.t()) + center_p2.t()

    return R, t


def rigid_transform_3d(p1, p2):
    center_p1 = np.mean(p1, axis=0, keepdims=True)
    center_p2 = np.mean(p2, axis=0, keepdims=True)

    pp1 = p1 - center_p1
    pp2 = p2 - center_p2

    H = np.matmul(pp1.T, pp2)

    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = np.matmul(-R, center_p1.T) + center_p2.T

    return R, t


def estimate_transform_target(p1, p2):
    assert len(p1) == len(p2)
    expand_dim = False
    if p1.shape[1] == 2:
        p1 = np.append(p1, np.zeros((p1.shape[0], 1)), 1)
        p2 = np.append(p2, np.zeros((p2.shape[0], 1)), 1)
        expand_dim = True
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    R, t = rigid_transform_3d(p1, p2)
    At = np.matmul(R, p1.T) + t
    if expand_dim:
        At = At[:-1, :]
    return At.T


def estimate_transform_target_pytorch(p1, p2):
    assert len(p1) == len(p2)
    expand_dim = False
    if p1.shape[1] == 2:
        p1 = torch.cat((p1, torch.zeros((p1.shape[0], 1)).float().to(p1.device)), dim=1)
        p2 = torch.cat((p2, torch.zeros((p2.shape[0], 1)).float().to(p1.device)), dim=1)
        expand_dim = True
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    R, t = rigid_transform_3d_pytorch(p1, p2)
    At = torch.mm(R, p1.t()) + t
    if expand_dim:
        At = At[:-1, :]
    return At.t()
