# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Latest update: Sep 1st 2023
# Licence: MIT


import os
import sys
import time
import torch
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_torch import VOT, VOTREG
import utils_torch as utils


# Generate data
N0 = 2000
K = 100
x, _ = sklearn.datasets.make_moons(n_samples=N0, noise=0.05, random_state=1)
y, labels = sklearn.datasets.make_moons(n_samples=K, noise=0.05, random_state=1)
y -= [0.5, 0.25]
x -= [0.5, 0.25]

y *= 0.5
x *= 0.5

theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
y = y.dot(R)


# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #
# ----- plot before ----- #
xmin, xmax, ymin, ymax = -1., 1., -1., 1.
cxs_base = np.array((utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED))
cys_base = np.array((utils.COLOR_BLUE, utils.COLOR_RED))
cys = cys_base[labels]
ys, xs = 15, 3

plt.figure(figsize=(12, 7))
plt.subplot(231)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/o reg before')

plt.scatter(x[:, 0], x[:, 1], s=xs, color=utils.COLOR_LIGHT_GREY)
for p, cy in zip(y, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy)


# ------- run WM -------- #
use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
y = torch.from_numpy(y).double().to(device)
x = torch.from_numpy(x).double().to(device)
labels = torch.from_numpy(labels).int().to(device)

vot = VOT(y.clone(), [x.clone()], label_y=labels, device=device, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
vot.cluster(lr=0.5, max_iter_y=1)
tock = time.time()
print('total running time: {0:.4f} seconds'.format(tock-tick))
cxs = cxs_base[vot.label_x[0]]


# ------ plot map ------- #
fig232 = plt.subplot(232)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/o reg map')

for p, p0 in zip(vot.y, vot.y_original):
    plt.plot([p[0], p0[0]], [p[1], p0[1]], color=np.append(utils.COLOR_LIGHT_GREY, 0.5), zorder=4)
for p, cy in zip(vot.y, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, facecolor='none', zorder=3)
for p, cy in zip(vot.y_original, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, zorder=2)

# ------ plot after ----- #
plt.subplot(233)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/o reg after')

for px, cx in zip(vot.x[0], cxs):
    plt.scatter(px[0], px[1], s=xs, color=cx, zorder=2)
for py, cy in zip(vot.y, cys):
    plt.scatter(py[0], py[1], s=ys, color=cy, facecolor='none', zorder=3)


# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #

# ------- run RWM ------- #
vot_reg = VOTREG(y.clone(), [x.clone()], label_y=labels, device=device, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
vot_reg.map(reg_type='transform', reg=1, max_iter_y=5, lr=0.5)
tock = time.time()
print("total running time : {0:.4f} seconds".format(tock-tick))

# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
print("[optional] distribute centroids into target domain...")
vot = VOT(vot_reg.y, vot_reg.x, label_y=labels, verbose=False)
vot.cluster(max_iter_y=1)
cxs = cxs_base[vot.label_x[0]]


# ------- plot map ------ #
plt.subplot(235)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/ reg map')

for p, p0 in zip(vot_reg.y, vot_reg.y_original):
    plt.plot([p[0], p0[0]], [p[1], p0[1]], color=np.append(utils.COLOR_LIGHT_GREY, 0.5), zorder=4)
for p, cy in zip(vot_reg.y, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, facecolor='none', zorder=3)
for p, cy in zip(vot_reg.y_original, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, zorder=2)

# ------ plot after ----- #
plt.subplot(236)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/ reg after')

for px, cx in zip(vot_reg.x[0], cxs):
    plt.scatter(px[0], px[1], s=xs, color=cx, zorder=2)
for py, cy in zip(vot_reg.y, cys):
    plt.scatter(py[0], py[1], s=ys, color=cy, facecolor='none', zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
plt.savefig("transform_torch.png")
# plt.show()
