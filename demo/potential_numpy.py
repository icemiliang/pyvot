# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: July 6th 2019

"""
===========================================
       Regularized Wasserstein Means
===========================================

This demo shows that regularizing the centroids by using class labels
and pairwise distances can benefit domain adaptation applications.

Predicted labels of the empirical samples come from the centroids.
It is equivalent to 1NN w.r.t. the power Euclidean distance.
"""
from __future__ import print_function
from __future__ import division
# import non-vot stuffs
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
# import vot stuffs
from vot_numpy import Vot, VotReg
import utils

# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #
data_p = np.loadtxt('data/p.csv', delimiter=",")
data_e = np.loadtxt('data/e.csv', delimiter=",")

# ------- run WM -------- #
vot = Vot(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0], verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
_, pred_label_e = vot.cluster(max_iter_p=5)
tock = time.time()
print("total running time : {0:g} seconds".format(tock-tick))

# ----- plot before ----- #
p_coor_before = vot.data_p_original
plt.figure(figsize=(12, 8))
xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0

cp = [utils.color_blue, utils.color_red]
cp = [cp[int(label)] for label in vot.label_p]
plt.subplot(231); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg before')
plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)

# ------ plot map ------- #
p_coor_after = np.copy(vot.data_p)
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[int(label)] for label in pred_label_e]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[int(label)] for label in vot.label_p]
plt.subplot(233); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg after')
plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #

# ------- run RWM ------- #
data_p = np.loadtxt('data/p.csv', delimiter=",")
data_e = np.loadtxt('data/e.csv', delimiter=",")

vot_reg = VotReg(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0], verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
vot_reg.cluster(reg_type='potential', reg=0.01, max_iter_p=5)
tock = time.time()
print("total running time : {0:g} seconds".format(tock-tick))

# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
print("[optional] distribute centroids into target domain...")
_, pred_label_e = vot_reg.cluster(max_iter_p=1)

# ------- plot map ------ #
p_coor_after = np.copy(vot_reg.data_p)
cp = [utils.color_blue, utils.color_red]
cp = [cp[int(label)] for label in vot_reg.label_p]
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig235 = plt.subplot(235); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/ reg map')
fig235.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[int(label)] for label in pred_label_e]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[int(label)] for label in vot_reg.label_p]
plt.subplot(236); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/ reg after')
plt.scatter(vot.data_e[:, 0], vot_reg.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("rwm_potential.png")
plt.show()
