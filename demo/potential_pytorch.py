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
from vot_pytorch import Vot, VotReg
import utils
import torch


# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #

# ------- run WM -------- #

data_p = np.loadtxt('data/p.csv', delimiter=",")
data_e = np.loadtxt('data/e.csv', delimiter=",")

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
data_p = torch.from_numpy(data_p).double().to(device)
data_e = torch.from_numpy(data_e).double().to(device)

ot = Vot(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0], device=device, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
_, e_predict = ot.cluster(0.5, max_iter_h=3000, max_iter_p=5)
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))


ot.data_p = ot.data_p.detach().cpu().numpy()
ot.data_e = ot.data_e.cpu().numpy()
ot.data_p_original = ot.data_p_original.cpu().numpy()
ot.label_p = ot.label_p.int().cpu().numpy()
ot.label_e = ot.label_e.int().cpu().numpy()
e_predict = e_predict.int().cpu().numpy()

# ----- plot before ----- #

p_coor_before = ot.data_p_original
plt.figure(figsize=(12, 8))

cp_base = [utils.color_blue, utils.color_red]
cp = [cp_base[label] for label in ot.label_p]
plt.subplot(231); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('w/o reg before')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)

# ------ plot map ------- #
p_coor_after = np.copy(ot.data_p)
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('w/o reg map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
ce_base = [utils.color_light_blue, utils.color_light_red]
cp_base = [utils.color_dark_blue, utils.color_red]
ce = [ce_base[label] for label in e_predict]
cp = [cp_base[label] for label in ot.label_p]
plt.subplot(233); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('w/o reg after')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #


# ------- run RWM ------- #
data_p = np.loadtxt('data/p.csv', delimiter=",")
data_p = torch.from_numpy(data_p).double().to(device)
ot_reg = VotReg(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0],  device=device, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
_, e_predict = ot_reg.cluster(reg_type=1, reg=0.1, max_iter_h=3000, max_iter_p=5)  # 0: w/o regularization
tock = time.time()
print("total running time : {} seconds".format(tock-tick))

# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
print("[optional] distribute centroids into target domain...")
ot_reg.cluster(0, max_iter_p=1)

ot_reg.data_p = ot_reg.data_p.detach().cpu().numpy()
ot_reg.data_e = ot_reg.data_e.cpu().numpy()
ot_reg.data_p_original = ot_reg.data_p_original.cpu().numpy()
ot_reg.label_p = ot_reg.label_p.int().cpu().numpy()
ot_reg.label_e = ot_reg.label_e.int().cpu().numpy()
e_predict = e_predict.int().cpu().numpy()


# ------- plot map ------ #
p_coor_after = np.copy(ot_reg.data_p)
cp_base = [utils.color_blue, utils.color_red]
cp = [cp_base[label] for label in ot.label_p]
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig235 = plt.subplot(235); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w/ reg map')
fig235.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
cp_base = [utils.color_dark_blue, utils.color_red]
ce = [ce_base[label] for label in e_predict]
cp = [cp_base[label] for label in ot_reg.label_p]
plt.subplot(236); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('w/ reg after')
plt.scatter(ot_reg.data_e[:, 0], ot_reg.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("rwm_potential.png")
plt.show()
