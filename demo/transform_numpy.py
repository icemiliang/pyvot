# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Aug 11th 2019
# Licence: MIT

"""
===========================================
       Regularized Wasserstein Means
===========================================

This demo shows that regularizing the centroids by using class labels
and pairwise distances can benefit domain adaptation applications.

Predicted labels of the empirical samples come from the centroids.
It is equivalent to 1NN w.r.t. the power Euclidean distance.
"""

import os
import sys
import time
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import Vot, VotReg
import utils


# Generate data
Ne = 2000
Np = 100
data_e, label_e = sklearn.datasets.make_moons(n_samples=Ne, noise=0.05, random_state=1)
data_p, label_p = sklearn.datasets.make_moons(n_samples=Np, noise=0.05, random_state=1)
data_p -= [0.5, 0.25]
data_e -= [0.5, 0.25]

data_p *= 0.5
data_e *= 0.5

theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
data_p = data_p.dot(R)

data_p1 = data_p.copy()
data_e1 = data_e.copy()

# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #

# ------- run WM -------- #
vot = Vot(data_p, data_e, label_p, label_e, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
_, pred_label_e = vot.cluster(0.5, max_iter_p=5)
tock = time.time()
print("total running time : {0:.4f} seconds".format(tock-tick))

# ----- plot before ----- #
plt.figure(figsize=(12, 7))
xmin, xmax, ymin, ymax = -1.0, 1.0, -.75, .75

plt.subplot(231)
cp_base = np.array([utils.COLOR_BLUE, utils.COLOR_RED])
cp = cp_base[vot.label_p.astype(np.int), :]
utils.plot_otsamples(vot.data_p_original, vot.data_e, color_p=cp, title='w/o reg before',
                     xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# ------ plot map ------- #
fig232 = plt.subplot(232)
cp_base = [utils.COLOR_BLUE, utils.COLOR_RED]
cp = np.array([cp_base[int(label)] for label in vot.label_p])
utils.plot_otmap(vot.data_p_original, vot.data_p, fig232, color=cp, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 title='w/o reg map', facecolor_after='none')

# ------ plot after ----- #
plt.subplot(233)
ce = np.array([utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED])[pred_label_e.astype(np.int), :]
cp = np.array([utils.COLOR_DARK_BLUE, utils.COLOR_RED])[vot.label_p.astype(np.int), :]
utils.plot_otsamples(vot.data_p, vot.data_e, size_p=30, marker_p='o', color_p=cp, color_e=ce,
                     xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title='w/o reg after', facecolor_p='none')


# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #

# ------- run RWM ------- #
vot_reg = VotReg(data_p1, data_e1, label_p, label_e, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
_, pred_label_e = vot_reg.cluster(reg_type='transform', reg=20, max_iter_p=5)
tock = time.time()
print("total running time : {0:.4f} seconds".format(tock-tick))

# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
print("[optional] distribute centroids into target domain...")
vot_reg.cluster(max_iter_p=1)

# ------- plot map ------ #
cp_base = [utils.COLOR_BLUE, utils.COLOR_RED]
cp = np.array([cp_base[int(label)] for label in vot_reg.label_p])
fig235 = plt.subplot(235)
utils.plot_otmap(vot_reg.data_p_original, vot_reg.data_p, fig235, color=cp, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                 title='w/ reg map', facecolor_after='none')

# ------ plot after ----- #
plt.subplot(236)
ce = np.array([utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED])[pred_label_e.astype(np.int), :]
cp = np.array([utils.COLOR_DARK_BLUE, utils.COLOR_RED])[vot_reg.label_p.astype(np.int), :]
utils.plot_otsamples(vot_reg.data_p, vot_reg.data_e, size_p=30, marker_p='o', color_p=cp, color_e=ce,
                     xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title='w/ reg after', facecolor_p='none')

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("rwm_transform.png")
plt.show()
