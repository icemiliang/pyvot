# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Aug 11th 2019
# Licence: MIT

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VOT, VOTREG
import utils

# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #
N0 = 1000
mean1, cov1 = [-0.5, 0.25], [[0.02, 0], [0, 0.02]]
mean2, cov2 = [ 0.5, 0.25], [[0.02, 0], [0, 0.02]]
x11, x12 = np.random.multivariate_normal(mean1, cov1, N0).T
x21, x22 = np.random.multivariate_normal(mean2, cov2, N0).T
x = np.concatenate([np.stack((x11, x12), axis=1), np.stack((x21, x22), axis=1)], axis=0).clip(-0.99, 0.99)

K = 50
mean3, cov3 = [0.0,  0.0], [[0.02, 0], [0, 0.02]]
mean4, cov4 = [0.5, -0.5], [[0.02, 0], [0, 0.02]]
y11, y12 = np.random.multivariate_normal(mean3, cov3, K).T
y21, y22 = np.random.multivariate_normal(mean4, cov4, K).T
y = np.concatenate((np.stack((y11, y12), axis=1), np.stack((y21, y22), axis=1)), axis=0).clip(-0.99, 0.99)
labels = np.concatenate((np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)), axis=0)

# ----- plot before ----- #
xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
cxs_base = np.array((utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED))
cys_base = np.array((utils.COLOR_BLUE, utils.COLOR_RED))
cys = cys_base[labels]
ys, xs = 15, 3

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/o reg before')

plt.scatter(x[:, 0], x[:, 1], s=xs, color=utils.COLOR_LIGHT_GREY)
for p, cy in zip(y, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy)


# ------- run WM -------- #
vot = VOT(y.copy(), [x.copy()], label_y=labels, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
vot.cluster(max_iter_y=1)
tock = time.time()
print("total running time : {0:g} seconds".format(tock-tick))
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
vot_reg = VOTREG(y.copy(), [x.copy()], label_y=labels, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
vot_reg.map(reg_type='potential', reg=0.01, max_iter_y=5)
tock = time.time()
print("total running time : {0:g} seconds".format(tock-tick))
# cxs = cxs_base[vot_reg.label_x[0]]

# Compute OT one more time to disperse the centroids into the empirical domain.
# This does not change the correspondence but can give better visual.
# This is optional.
print("[optional] distribute centroids into target domain...")
vot = VOT(vot_reg.y, vot_reg.x, label_y=labels, verbose=False)
vot.cluster(max_iter_y=1)
cxs = cxs_base[vot.label_x[0]]


# ------ plot map ------- #
plt.subplot(235)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/ reg map')

for p, p0 in zip(vot.y, vot_reg.y_original):
    plt.plot([p[0], p0[0]], [p[1], p0[1]], color=np.append(utils.COLOR_LIGHT_GREY, 0.5), zorder=4)
for p, cy in zip(vot.y, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, facecolor='none', zorder=3)
for p, cy in zip(vot_reg.y_original, cys):
    plt.scatter(p[0], p[1], s=ys, color=cy, zorder=2)


# ------ plot after ----- #
plt.subplot(236)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True)
plt.title('w/ reg after')

for px, cx in zip(vot.x[0], cxs):
    plt.scatter(px[0], px[1], s=xs, color=cx, zorder=2)
for py, cy in zip(vot.y, cys):
    plt.scatter(py[0], py[1], s=ys, color=cy, facecolor='none', zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
plt.savefig("rwm_potential.png")
plt.show()
