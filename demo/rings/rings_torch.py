# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from math import pi, cos, sin
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_torch import VWB


n = 500
t = np.linspace(0, 2 * pi, n)
N = 10
linewidth = 2

xmin = -1
xmax = 1
ymin = -1
ymax = 1

np.random.seed(19)


def get_ellips(u, v, a, b, theta1, theta2):
    ell_in = np.array([a*np.cos(t), b*np.sin(t)])

    r1 = np.array([[cos(theta1), -sin(theta1)], [sin(theta1), cos(theta1)]])
    r2 = np.array([[cos(theta2), -sin(theta2)], [sin(theta2), cos(theta2)]])

    scaleu = np.random.normal(0.5, 0.1)
    scalev = np.random.normal(0.5, 0.1)
    ell_in[0, :] *= scaleu
    ell_in[1, :] *= scalev
    ell_out = np.matmul(r1, ell_in)
    ell_in = np.matmul(r2, ell_in)

    # u = 0.2 * np.random.normal(size=[n]) + u
    # v = 0.2 * np.random.normal(size=[n]) + v

    # ell_out *= scale
    ell_in[0, :] *= scaleu
    ell_in[1, :] *= scalev

    ell_in_orig = ell_in.copy()
    ell_out_orig = ell_out.copy()

    ell_out[0, :] += u
    ell_out[1, :] += v
    ell_in[0, :] += u
    ell_in[1, :] += v

    return ell_in, ell_out, ell_in_orig, ell_out_orig


uu = np.random.uniform(-0.5, 0.5, N)
vv = np.random.uniform(-0.5, 0.5, N)
aa = np.random.uniform(0.5, 1., N)
bb = np.random.uniform(0.5, 1., N)
rr1 = np.random.uniform(0, pi, N)
rr2 = np.random.uniform(0, pi, N)

ell_all = []
ell_all_in = []
ell_all_out = []
ell_all_orig = []
ell_all_orig_in = []
ell_all_orig_out = []
cm = plt.get_cmap('hsv')
s = np.size(t)
T = np.linspace(0, 1, s) ** 2

k = 1

for u, v, a, b, r1, r2 in zip(uu, vv, aa, bb, rr1, rr2):
    ell_in, ell_out, ell_in_orig, ell_out_orig = get_ellips(u, v, a, b, r1, r2)
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
    for i in range(s - 1):
        ax.plot(ell_in[0, i:i + 2], ell_in[1, i:i + 2])
        ax.plot(ell_out[0, i:i + 2], ell_out[1, i:i + 2])

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(color='lightgray', linestyle='--')
    # plt.savefig("test" + str(k) + ".svg", bbox_inches='tight')
    plt.savefig("test" + str(k) + ".png", dpi=300, bbox_inches='tight')
    ell_all.append(np.append(ell_in, ell_out, axis=1))
    ell_all_in.append(ell_in)
    ell_all_out.append(ell_out)
    ell_all_orig_in.append(ell_in_orig)
    ell_all_orig_out.append(ell_out_orig)
    k += 1


fig = plt.figure(2, figsize=(2, 2))
ax = fig.add_subplot(111)
ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
for ell_in, ell_out in zip(ell_all_in, ell_all_out):
    for i in range(s - 1):
        ax.plot(ell_in[0, i:i + 2], ell_in[1, i:i + 2])
        ax.plot(ell_out[0, i:i + 2], ell_out[1, i:i + 2])

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
# plt.savefig("euclidean.svg", bbox_inches='tight')
plt.savefig("euclidean.png", dpi=300, bbox_inches='tight')


fig = plt.figure(3, figsize=(2, 2))
ax = fig.add_subplot(111)
ax.set_prop_cycle(cycler(color=[cm(1. * i / (s - 1)) for i in range(s - 1)]))
for ell_in, ell_out in zip(ell_all_orig_in, ell_all_orig_out):
    for i in range(s - 1):
        ax.plot(ell_in[0, i:i + 2], ell_in[1, i:i + 2])
        ax.plot(ell_out[0, i:i + 2], ell_out[1, i:i + 2])

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
# plt.savefig("euclidean_center.svg", bbox_inches='tight')
plt.savefig("euclidean_center.png", dpi=300, bbox_inches='tight')

# VWB
use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

ell_all = [torch.from_numpy(ell).t() for ell in ell_all]

vwb = VWB(ell_all[0][0:-1:5, :], ell_all, device=device, verbose=False)
output = vwb.cluster(lr=0.5, max_iter_h=3000, max_iter_p=1)
e_idx = output['idx']

fig = plt.figure(4, figsize=(2, 2))
ax = fig.add_subplot(111)

plt.grid(True)
for i in range(N):
    plt.plot(vwb.data_e[i][:, 0], vwb.data_e[i][:, 1], linewidth=5, c='lightgray')
size = vwb.data_p.shape[0] // 2

ax.set_prop_cycle(cycler(color=[cm(1. * i / (size - 1)) for i in range(size - 1)]))
ell1 = vwb.data_p[:size, :]
ell2 = vwb.data_p[size:, :]
for i in range(s - 1):
    ax.plot(ell1[i:i + 2, 0], ell1[i:i + 2, 1])
    ax.plot(ell2[i:i + 2, 0], ell2[i:i + 2, 1])


plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.axis("tight")
# plt.savefig("test_vwb.svg", bbox_inches='tight')
plt.savefig("test_vwb.png", dpi=300, bbox_inches='tight')
