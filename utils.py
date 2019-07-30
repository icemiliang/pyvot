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


def assert_boundary(data):
    assert data.max() < 1, warnings.warn("Data out of boundary (-1, 1).")
    assert -1 < data.min(), warnings.warn("Data out of boundary (-1, 1).")


def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    buf = fig2data(fig)
    w, h = buf.shape[0], buf.shape[1]
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def random_sample(num, dim, sampling='square'):
    """ randomly sample the area with dirac measures
        area boundary: [-0.99, 0.99] in each dimension
    """
    data = None

    if num * dim > 1e8:
        warnings.warn("Sampling the area will take too much memory.")
    if sampling == 'square':
        data = np.random.random((num, dim)) * 1.98 - 1
    elif sampling == 'disk':
        r = np.random.uniform(low=0, high=0.99, size=num)  # radius
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)  # angle
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        data = np.concatenate((x[:, None], y[:, None]), axis=1)
    elif sampling == 'circle':
        r = np.random.uniform(low=0.8, high=0.99, size=num)  # radius
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)  # angle
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        data = np.concatenate((x[:, None], y[:, None]), axis=1)
    elif sampling == 'gaussian' or sampling == 'gauss':
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
    plt.scatter(data[:, 0], data[:, 1], s=1, marker='o', color=color(idx))
    return fig


def rigid_transform_3d_pytorch(p1, p2):
    center_p1 = torch.mean(p1, dim=0, keepdim=True)
    center_p2 = torch.mean(p2, dim=0, keepdim=True)

    pp1 = p1 - center_p1
    pp2 = p2 - center_p2

    h = torch.mm(pp1.t(), pp2)
    u, _, v = torch.svd(h)
    r = torch.mm(v.t(), u.t())

    # reflection
    if np.linalg.det(r.cpu().numpy()) < 0:
        v[2, :] *= -1
        r = torch.mm(v.t(), u.t())

    t = torch.mm(-r, center_p1.t()) + center_p2.t()

    return r, t


def rigid_transform_3d_numpy(p1, p2):
    center_p1 = np.mean(p1, axis=0, keepdims=True)
    center_p2 = np.mean(p2, axis=0, keepdims=True)

    pp1 = p1 - center_p1
    pp2 = p2 - center_p2

    h = np.matmul(pp1.T, pp2)
    u, _, v = np.linalg.svd(h)
    r = np.matmul(v.T, u.T)

    # reflection
    if np.linalg.det(r) < 0:
        v[2, :] *= -1
        r = np.matmul(v.T, u.T)

    t = np.matmul(-r, center_p1.T) + center_p2.T

    return r, t


def estimate_transform_target(p1, p2):
    assert len(p1) == len(p2)
    expand_dim = False
    if p1.shape[1] == 2:
        p1 = np.append(p1, np.zeros((p1.shape[0], 1)), 1)
        p2 = np.append(p2, np.zeros((p2.shape[0], 1)), 1)
        expand_dim = True
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    # TODO downsample points if too many

    r, t = rigid_transform_3d_numpy(p1, p2)
    At = np.matmul(r, p1.T) + t
    if expand_dim:
        At = At[:-1, :]
    return At.T


def estimate_transform_target_pytorch(p1, p2):
    assert len(p1) == len(p2)

    # Mask out nan which came from empty clusters
    mask = torch.isnan(p2).any(dim=1)

    expand_dim = False
    if p1.shape[1] == 2:
        p1 = torch.cat((p1, torch.zeros((p1.shape[0], 1)).float().to(p1.device)), dim=1)
        p2 = torch.cat((p2, torch.zeros((p2.shape[0], 1)).float().to(p2.device)), dim=1)
        expand_dim = True
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    # TODO downsample points if too many
    p11 = p1[~mask]
    p22 = p2[~mask]

    r, t = rigid_transform_3d_pytorch(p11, p22)
    At = torch.mm(r, p1.t()) + t
    if expand_dim:
        At = At[:-1, :]
    return At.t()
