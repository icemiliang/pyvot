# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: July 6th 2019

"""
===========================================
       Regularized Wasserstein Means
===========================================

This demo shows that regularizing the centroids by using prior
geometric transformation can benefit domain adaptation applications.

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
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.collections as mc
# import vot stuffs
from vot_pytorch import Vot
import utils
import torch


# Generate data
num_e = 2000
num_p = 100
data_e, label_e = sklearn.datasets.make_moons(n_samples=num_e, noise=0.05, random_state=1)
data_p, label_p = sklearn.datasets.make_moons(n_samples=num_p, noise=0.05, random_state=1)
data_p = (data_p - [0.5, 0.25]) / 2
data_e = (data_e - [0.5, 0.25]) / 2

data_p = np.clip(data_p, -0.99, 0.99)
data_e = np.clip(data_e, -0.99, 0.99)

theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
data_p = data_p.dot(R)
label_p = label_p.transpose()

data_p1 = data_p.copy()
data_e1 = data_e.copy()


# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #

# ------- run WM -------- #
use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
data_p = torch.from_numpy(data_p)
data_e = torch.from_numpy(data_e)
label_p = torch.from_numpy(label_p)
label_e = torch.from_numpy(label_e)

ot = Vot(data_p=data_p, data_e=data_e,
         label_p=label_p, label_e=label_e,
         device=device, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
ot.cluster(0, max_iter_h=5000, max_iter_p=5, lr=0.2, lr_decay=500)  # 0: w/o regularization
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))

# ----- plot before ----- #
ot.data_p = ot.data_p.detach().cpu().numpy()
ot.data_e = ot.data_e.cpu().numpy()
ot.data_p_original = ot.data_p_original.cpu().numpy()
ot.data_e_original = ot.data_e_original.cpu().numpy()
ot.label_p = ot.label_p.int().cpu().numpy()
ot.label_e = ot.label_e.int().cpu().numpy()
ot.e_predict = ot.e_predict.int().cpu().numpy()

p_coor_before = ot.data_p_original
plt.figure(figsize=(12, 7))
xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0

cp = [utils.color_blue, utils.color_red]
cp = [cp[label] for _, label in np.ndenumerate(ot.label_p)]
plt.subplot(231); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg before')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)

# ------ plot map ------- #
p_coor_after = np.copy(ot.data_p)
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[label] for _, label in np.ndenumerate(le)]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[label] for _, label in np.ndenumerate(ot.label_p)]
plt.subplot(233); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/o reg after')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)


# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #

# ------- run RWM ------- #
data_p1 = torch.from_numpy(data_p1)
data_e1 = torch.from_numpy(data_e1)
ot_reg = Vot(data_p=data_p1, data_e=data_e1,
             label_p=label_p, label_e=label_e,
             device=device, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
ot_reg.cluster(reg_type='transform', reg=20, max_iter_h=5000, lr=0.2, max_iter_p=5)
tock = time.time()
print("total running time : {} seconds".format(tock-tick))

# Compute vanilla OT one more time to disperse the centroids into the empirical domain.
# This optional step almost does not change the correspondence but gives better positions.
print("[optional] distribute centroids into target domain...")
ot_reg.cluster(0, max_iter_h=5000, lr=0.1, max_iter_p=1)


# ----- plot before ----- #
ot_reg.data_p = ot_reg.data_p.detach().cpu().numpy()
ot_reg.data_e = ot_reg.data_e.cpu().numpy()
ot_reg.data_p_original = ot_reg.data_p_original.cpu().numpy()
ot_reg.data_e_original = ot_reg.data_e_original.cpu().numpy()
ot_reg.label_p = ot_reg.label_p.int().cpu().numpy()
ot_reg.label_e = ot_reg.label_e.int().cpu().numpy()
ot_reg.e_predict = ot_reg.e_predict.int().cpu().numpy()


cp = [utils.color_blue, utils.color_red]
cp = [cp[label] for _, label in np.ndenumerate(ot_reg.label_p)]
plt.subplot(234); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/ reg before')
plt.scatter(ot_reg.data_e[:, 0], ot_reg.data_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)

# ------- plot map ------ #
p_coor_after = np.copy(ot_reg.data_p)
ot_map = [[tuple(p1), tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig235 = plt.subplot(235); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/ reg map')
fig235.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='o', color=cp, zorder=3)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot_reg.e_predict)
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[label] for _, label in np.ndenumerate(le)]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[label] for _, label in np.ndenumerate(ot_reg.label_p)]
plt.subplot(236); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.grid(True); plt.title('w/ reg after')
plt.scatter(ot.data_e[:, 0], ot_reg.data_e[:, 1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:, 0], p_coor_after[:, 1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("rwm_potential.png")
plt.show()
