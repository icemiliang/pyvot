# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 25th 2020
# Licence: MIT

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from math import pi, cos, sin
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VOT


SAVE_SVG = False
SAVE_PNG = False
PLOT_FIGURE = False
SAVE_OR_PLOT_FIGURE = SAVE_PNG or SAVE_SVG or PLOT_FIGURE

n = 500
t = np.linspace(0, 2 * pi, n)
N = 10
linewidth = 2

xmin, xmax, ymin, ymax = -1, 1, -1, 1

np.random.seed(19)


def get_rings(u, v, a, b, inner_theta, outer_theta):
    inner_ring = np.array([a*np.cos(t), b*np.sin(t)])

    r1 = np.array([[cos(inner_theta), -sin(inner_theta)], [sin(inner_theta), cos(inner_theta)]])
    r2 = np.array([[cos(outer_theta), -sin(outer_theta)], [sin(outer_theta), cos(outer_theta)]])

    scaleu = np.random.normal(0.5, 0.1)
    scalev = np.random.normal(0.5, 0.1)

    inner_ring[0, :] *= scaleu
    inner_ring[1, :] *= scalev
    outer_ring = np.matmul(r1, inner_ring)
    inner_ring = np.matmul(r2, inner_ring)

    inner_ring[0, :] *= scaleu
    inner_ring[1, :] *= scalev

    inner_ring_centered = inner_ring.copy()
    outer_ring_centered = outer_ring.copy()

    outer_ring[0, :] += u
    outer_ring[1, :] += v
    inner_ring[0, :] += u
    inner_ring[1, :] += v

    return inner_ring, outer_ring, inner_ring_centered, outer_ring_centered


uu = np.random.uniform(-0.5, 0.5, N)
vv = np.random.uniform(-0.5, 0.5, N)
aa = np.random.uniform(0.5, 1., N)
bb = np.random.uniform(0.5, 1., N)
rr1 = np.random.uniform(0, pi, N)
rr2 = np.random.uniform(0, pi, N)

all_rings = []
all_inner_rings = []
all_outer_rings = []
ell_all_orig = []
all_inner_rings_centered = []
all_outer_rings_centered = []

cm = plt.get_cmap('hsv')
s = np.size(t)
T = np.linspace(0, 1, s) ** 2

figure_id = 0

for u, v, a, b, r1, r2, i in zip(uu, vv, aa, bb, rr1, rr2, range(N)):
    inner_ring, outer_ring, inner_ring_center, outer_ring_center = get_rings(u, v, a, b, r1, r2)

    all_rings.append(np.append(inner_ring, outer_ring, axis=1))
    all_inner_rings.append(inner_ring)
    all_outer_rings.append(outer_ring)
    all_inner_rings_centered.append(inner_ring_center)
    all_outer_rings_centered.append(outer_ring_center)

    if SAVE_OR_PLOT_FIGURE:
        fig = plt.figure(figure_id, figsize=(2, 2))
        figure_id += 1
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
        for i in range(s - 1):
            ax.plot(inner_ring[0, i: i + 2], inner_ring[1, i: i + 2])
            ax.plot(outer_ring[0, i: i + 2], outer_ring[1, i: i + 2])

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        plt.grid(color='lightgray', linestyle='--')

        if SAVE_SVG:
            plt.savefig("rings_" + str(i) + ".svg", bbox_inches='tight')
        if SAVE_PNG:
            plt.savefig("rings_" + str(i) + ".png", dpi=300, bbox_inches='tight')


if SAVE_OR_PLOT_FIGURE:
    fig = plt.figure(figure_id, figsize=(2, 2))
    figure_id += 1
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
    for inner_ring, outer_ring in zip(all_inner_rings, all_outer_rings):
        for i in range(s - 1):
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
    ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
    for inner_ring, outer_ring in zip(all_inner_rings_centered, all_outer_rings_centered):
        for i in range(s - 1):
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

vwb = VOT(all_rings[0][0:-1:5, :], all_rings, verbose=False)
tick = time.clock()
vwb.cluster(lr=0.5, max_iter_h=3000, max_iter_y=1)
tock = time.clock()
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
    for i in range(s - 1):
        ax.plot(inner_ring[i:i + 2, 0], inner_ring[i:i + 2, 1])
        ax.plot(outer_ring[i:i + 2, 0], outer_ring[i:i + 2, 1])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])


if SAVE_SVG:
    plt.savefig("vot.svg", bbox_inches='tight')
if SAVE_PNG:
    plt.savefig("vot.png", dpi=300, bbox_inches='tight')

if PLOT_FIGURE:
    plt.show()
