# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT

import numpy as np
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import torch


COLOR_BLUE = [0.12, 0.56, 1]
COLOR_LIGHT_BLUE = [0.5, 0.855, 1]
COLOR_DARK_BLUE = [0.05, 0.28, 0.63]

COLOR_RED = [0.8, 0.22, 0]
COLOR_LIGHT_RED = [1.0, 0.54, 0.5]

COLOR_LIGHT_GREY = [0.7, 0.7, 0.7]
COLOR_GREY = [0.5, 0.5, 0.5]


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
        data = torch.rand((num, dim)) * 1.98 - 1
    elif sampling == 'disk' or sampling == 'circle':
        if sampling == 'disk':
            r = torch.rand(num)  # radius
        else:
            r = torch.rand(num) * 0.29 + 0.7  # radius
        theta = torch.rand(num) * 2 * np.pi  # angle
        x = torch.sqrt(r) * torch.cos(theta)
        y = torch.sqrt(r) * torch.sin(theta)
        data = torch.cat((x[:, None], y[:, None]), dim=1)
    elif sampling == 'gaussian' or sampling == 'gauss':
        mean = torch.tensor([0, 0])
        cov = torch.tensor([[.1, 0], [0, .1]])
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        data = m.sample(torch.Size([num])).clip(-0.99, 0.99)

    label = -torch.ones(num, dtype=torch.int)
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


def rigid_transform_3d(p1, p2):
    center_p1 = torch.mean(p1, dim=0, keepdim=True)
    center_p2 = torch.mean(p2, dim=0, keepdim=True)

    pp1 = p1 - center_p1
    pp2 = p2 - center_p2

    h = torch.mm(pp1.t(), pp2)
    u, _, v = torch.svd(h)
    r = torch.mm(v, u.T)

    # reflection
    if torch.linalg.det(r) < 0:
        v[2, :] *= -1
        r = torch.mm(v.t(), u.t())

    t = torch.mm(-r, center_p1.t()) + center_p2.t()

    return r, t


def estimate_transform(p1, p2):
    assert len(p1) == len(p2)
    if p1.shape[1] == 2:
        p1 = torch.hstack([p1, torch.zeros((p1.shape[0], 1))])
        p2 = torch.hstack([p2, torch.zeros((p2.shape[0], 1))])
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    r, t = rigid_transform_3d(p1, p2)

    return r, t


def estimate_inverse_transform(p1, p2):
    return estimate_transform(p2, p1)


def estimate_transform_target(p1, p2):
    assert len(p1) == len(p2)

    # Mask out nan which came from empty clusters
    mask = torch.isnan(p2).any(dim=1)

    expand_dim = False
    if p1.shape[1] == 2:
        p1 = torch.cat((p1, torch.zeros((p1.shape[0], 1)).double().to(p1.device)), dim=1)
        p2 = torch.cat((p2, torch.zeros((p2.shape[0], 1)).double().to(p2.device)), dim=1)
        expand_dim = True
    elif p1.shape[1] != 3:
        raise Exception("expected 2d or 3d points")

    p11 = p1[~mask]
    p22 = p2[~mask]

    r, t = rigid_transform_3d(p11, p22)
    print(r)
    print(t)
    At = torch.mm(r, p1.t().clone()) + t
    if expand_dim:
        At = At[:-1, :]
    return At.t()


def scatter_otsamples(data_p, data_e=None, color_y=None, color_x=None, title="", grid=True, marker_p='o', marker_e='.',
                      facecolor_p=None, size_p=20, size_e=20, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, nop=False):
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(grid)
    plt.title(title)

    if data_e is not None:
        if color_x is not None:
            assert len(color_x) == 3 \
                   or (color_x.ndim == 2 and color_x.shape[0] == data_e.shape[0] and (color_x.shape[1] == 3 or color_x.shape[1] == 4))
        else:
            color_x = COLOR_LIGHT_GREY
        plt.scatter(data_e[:, 0], data_e[:, 1], s=size_e, marker=marker_e, color=color_x, zorder=2)

    if color_y is not None:
        assert len(color_y) == 3 \
               or (color_y.ndim == 2 and color_y.shape[0] == data_p.shape[0] and (color_y.shape[1] == 3 or color_y.shape[1] == 4))
    else:
        color_y = COLOR_RED
    if nop == False:
        if facecolor_p == 'none':
            plt.scatter(data_p[:, 0], data_p[:, 1], s=size_p, marker=marker_p, facecolors='none', linewidth=2, color=color_y, zorder=3)
        else:
            plt.scatter(data_p[:, 0], data_p[:, 1], s=size_p, marker=marker_p, linewidth=2, color=color_y, zorder=3)


def scatter_otsamples3D(data_p, data_e=None, color_p=None, color_e=None, title="", grid=True, marker_p='o', marker_e='.',
                      facecolor_p=None, size_p=20, size_e=20, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, nop=False):
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(grid)
    plt.title(title)

    if data_e is not None:
        if color_e is not None:
            assert len(color_e) == 3 \
                   or (color_e.ndim == 2 and color_e.shape[0] == data_e.shape[0] and (color_e.shape[1] == 3 or color_e.shape[1] == 4))
        else:
            color_e = COLOR_LIGHT_GREY
        plt.scatter(data_e[:, 0], data_e[:, 1], data_e[:, 1], s=size_e, marker=marker_e, color=color_e, zorder=2)

    if color_p is not None:
        assert len(color_p) == 3 \
               or (color_p.ndim == 2 and color_p.shape[0] == data_p.shape[0] and (color_p.shape[1] == 3 or color_p.shape[1] == 4))
    else:
        color_p = COLOR_RED
    if nop == False:
        if facecolor_p == 'none':
            plt.scatter(data_p[:, 0], data_p[:, 1], data_p[:, 1], s=size_p, marker=marker_p, facecolors='none', linewidth=2, color=color_p, zorder=3)
        else:
            plt.scatter(data_p[:, 0], data_p[:, 1], data_p[:, 1], s=size_p, marker=marker_p, linewidth=2, color=color_p, zorder=3)


def plot_otsamples(y, x=None, color_y=None, color_x=None, linewidth=2, title="", grid=True,
                   xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0):
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(grid)
    plt.title(title)

    if x is not None:
        if color_x is not None:
            assert len(color_x) == 3 \
                   or (color_x.ndim == 2 and color_x.shape[0] == x.shape[0] and (color_x.shape[1] == 3 or color_x.shape[1] == 4))
        else:
            color_x = COLOR_LIGHT_GREY
        plt.plot(x[:, 0], x[:, 1], color=color_x, zorder=2)

    if color_y is not None:
        assert len(color_y) == 3 \
               or (color_y.ndim == 2 and color_y.shape[0] == y.shape[0] and (color_y.shape[1] == 3 or color_y.shape[1] == 4))
    else:
        color_y = COLOR_RED
    if color_y is None:
        plt.plot(y[:, 0], y[:, 1], linewidth=linewidth, c='r', zorder=3)
    else:
        plt.plot(y[:, 0], y[:, 1], linewidth=linewidth, c=color_y, zorder=3)


def plot_otmap(data_before, data_after, plt_fig, color=None, title="", grid=True, marker='o', facecolor_after=None,
               facecolor_before=None,
               xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0):

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(grid)
    plt.title(title)

    ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(data_before.tolist(), data_after.tolist())]
    lines = mc.LineCollection(ot_map, colors=COLOR_LIGHT_GREY)
    fig = plt_fig
    fig.add_collection(lines)

    if color is not None:
        assert color.shape[0] == 3 \
               or (color.ndim == 2 and color.shape[0] == color.shape[0] and color.shape[1] == 3)
    else:
        color = COLOR_RED

    if facecolor_before == 'none':
        plt.scatter(data_before[:, 0], data_before[:, 1], marker=marker, facecolors='none', linewidth=2, color=color, zorder=2)
    else:
        plt.scatter(data_before[:, 0], data_before[:, 1], marker=marker, linewidth=2, color=color, zorder=2)

    if facecolor_after == 'none':
        plt.scatter(data_after[:, 0], data_after[:, 1], marker=marker, facecolors='none', linewidth=2, color=color, zorder=2)
    else:
        plt.scatter(data_after[:, 0], data_after[:, 1], marker=marker, linewidth=2, color=color, zorder=2)
