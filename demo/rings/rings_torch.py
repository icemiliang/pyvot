# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Latest update: Sep 1st 2023
# Licence: MIT

# import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import os
import sys
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_torch import VOT


SAVE_SVG = False
SAVE_PNG = True
PLOT_FIGURE = False
SAVE_OR_PLOT_FIGURE = SAVE_PNG or SAVE_SVG or PLOT_FIGURE

n = 500
t = torch.linspace(0, 2 * torch.pi, n)
N = 10
linewidth = 2

xmin, xmax, ymin, ymax = -1, 1, -1, 1

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

torch.manual_seed(19)


def get_rings(u, v, a, b, inner_theta, outer_theta):
    inner_ring = torch.stack((a*torch.cos(t), b*torch.sin(t)), dim=0)

    r1 = torch.Tensor([[torch.cos(inner_theta), -torch.sin(inner_theta)], [torch.sin(inner_theta), torch.cos(inner_theta)]])
    r2 = torch.Tensor([[torch.cos(outer_theta), -torch.sin(outer_theta)], [torch.sin(outer_theta), torch.cos(outer_theta)]])

    scaleu = torch.randn(1) * 0.1 + 0.5
    scalev = torch.randn(1) * 0.1 + 0.5

    inner_ring[0, :] *= scaleu
    inner_ring[1, :] *= scalev
    outer_ring = torch.matmul(r1, inner_ring)
    inner_ring = torch.matmul(r2, inner_ring)

    inner_ring[0, :] *= scaleu
    inner_ring[1, :] *= scalev

    inner_ring_centered = inner_ring.clone()
    outer_ring_centered = outer_ring.clone()

    outer_ring[0, :] += u
    outer_ring[1, :] += v
    inner_ring[0, :] += u
    inner_ring[1, :] += v

    return inner_ring, outer_ring, inner_ring_centered, outer_ring_centered


uu = torch.rand(N) - 0.5
vv = torch.rand(N) - 0.5
aa = torch.rand(N) / 2 + 0.5
bb = torch.rand(N) / 2 + 0.5
rr1 = torch.rand(N) * torch.pi
rr2 = torch.rand(N) * torch.pi

all_rings = []
all_inner_rings = []
all_outer_rings = []
ell_all_orig = []
all_inner_rings_centered = []
all_outer_rings_centered = []

cm = plt.get_cmap('hsv')
T = torch.linspace(0, 1, n) ** 2

figure_id = 0

for u, v, a, b, r1, r2, i, in zip(uu, vv, aa, bb, rr1, rr2, range(N)):
    inner_ring, outer_ring, inner_ring_center, outer_ring_center = get_rings(u, v, a, b, r1, r2)

    all_rings.append(torch.cat((inner_ring, outer_ring), dim=1))
    all_inner_rings.append(inner_ring)
    all_outer_rings.append(outer_ring)
    all_inner_rings_centered.append(inner_ring_center)
    all_outer_rings_centered.append(outer_ring_center)

    if SAVE_OR_PLOT_FIGURE:
        fig = plt.figure(figure_id, figsize=(2, 2))
        figure_id += 1
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(cycler(color=[cm(1. * j / (n - 1)) for j in range(n - 1)]))
        for j in range(n - 1):
            ax.plot(inner_ring[0, j: j + 2], inner_ring[1, j: j + 2])
            ax.plot(outer_ring[0, j: j + 2], outer_ring[1, j: j + 2])

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        plt.grid(color='lightgray', linestyle='--')

        if SAVE_SVG:
            plt.savefig("rings_" + str(i) + "_torch.svg", bbox_inches='tight')
        if SAVE_PNG:
            plt.savefig("rings_" + str(i) + "_torch.png", dpi=300, bbox_inches='tight')

if SAVE_OR_PLOT_FIGURE:
    fig = plt.figure(figure_id, figsize=(2, 2))
    figure_id += 1
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler(color=[cm(1. * i / (n - 1)) for i in range(n - 1)]))
    for inner_ring, outer_ring in zip(all_inner_rings, all_outer_rings):
        for i in range(n - 1):
            ax.plot(inner_ring[0, i: i + 2], inner_ring[1, i: i + 2])
            ax.plot(outer_ring[0, i: i + 2], outer_ring[1, i: i + 2])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    if SAVE_SVG:
        plt.savefig("euclidean.svg", bbox_inches='tight')
    if SAVE_PNG:
        plt.savefig("euclidean.png", dpi=300, bbox_inches='tight')


if SAVE_OR_PLOT_FIGURE:
    fig = plt.figure(figure_id, figsize=(2, 2))
    figure_id += 1
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler(color=[cm(1. * i / (n - 1)) for i in range(n - 1)]))
    for inner_ring, outer_ring in zip(all_inner_rings_centered, all_outer_rings_centered):
        for i in range(n - 1):
            ax.plot(inner_ring[0, i:i + 2], inner_ring[1, i:i + 2])
            ax.plot(outer_ring[0, i:i + 2], outer_ring[1, i:i + 2])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    if SAVE_SVG:
        plt.savefig("euclidean_center.svg", bbox_inches='tight')
    if SAVE_PNG:
        plt.savefig("euclidean_center.png", dpi=300, bbox_inches='tight')

all_rings = [ring.T for ring in all_rings]

# VWB


vwb = VOT(all_rings[0][0:-1:5, :], all_rings, verbose=False)
tick = time.process_time()
vwb.cluster(lr=0.5, max_iter_h=3000, max_iter_y=1)
tock = time.process_time()
print(tock - tick)

if SAVE_OR_PLOT_FIGURE:
    fig = plt.figure(figure_id, figsize=(2, 2))
    ax = fig.add_subplot(111)

    for i in range(N):
        plt.plot(vwb.x[i][:, 0], vwb.x[i][:, 1], linewidth=5, c='lightgray')
    size = vwb.y.shape[0] // 2

    ax.set_prop_cycle(cycler(color=[cm(1. * i / (size - 1)) for i in range(size - 1)]))
    inner_ring = vwb.y[:size, :]
    outer_ring = vwb.y[size:, :]
    for i in range(n - 1):
        ax.plot(inner_ring[i:i + 2, 0], inner_ring[i:i + 2, 1])
        ax.plot(outer_ring[i:i + 2, 0], outer_ring[i:i + 2, 1])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])


if SAVE_SVG:
    plt.savefig("vot_torch.svg", bbox_inches='tight')
if SAVE_PNG:
    plt.savefig("vot_torch.png", dpi=300, bbox_inches='tight')

if PLOT_FIGURE:
    plt.show()
