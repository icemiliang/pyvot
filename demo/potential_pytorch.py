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
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_pytorch import Vot, VotReg
import utils


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

vot = Vot(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0], device=device, verbose=False)
print("running Wasserstein clustering...")
tick = time.time()
_, pred_label_e = vot.cluster(0.5, max_iter_h=3000, max_iter_p=5)
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))


# vot.data_p = vot.data_p.detach().cpu().numpy()
# vot.data_e = vot.data_e.cpu().numpy()
# vot.data_p_original = vot.data_p_original.cpu().numpy()
# vot.label_p = vot.label_p.int().cpu().numpy()
# vot.label_e = vot.label_e.int().cpu().numpy()
# pred_label_e = pred_label_e.int().cpu().numpy()

# ----- plot before ----- #
plt.figure(figsize=(12, 8))
plt.subplot(231)
cp = np.array([utils.COLOR_BLUE, utils.COLOR_RED])[vot.label_p.int().cpu().numpy(), :]
utils.plot_otsamples(vot.data_p_original, vot.data_e, color_p=cp, title='w/o reg before')

# ------ plot map ------- #
fig232 = plt.subplot(232)
cp = np.array([utils.COLOR_BLUE, utils.COLOR_RED])[vot.label_p.int().cpu().numpy(), :]
utils.plot_otmap(vot.data_p_original, vot.data_p, fig232, color=cp, title='w/o reg map')

# ------ plot after ----- #
plt.subplot(233)
ce = np.array([utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED])[pred_label_e.int().cpu().numpy(), :]
cp = np.array([utils.COLOR_DARK_BLUE, utils.COLOR_RED])[vot.label_p.int().cpu().numpy(), :]
utils.plot_otsamples(vot.data_p, vot.data_e, color_p=cp, color_e=ce, title='w/o reg after')


# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #


# ------- run RWM ------- #
data_p = np.loadtxt('data/p.csv', delimiter=",")
data_p = torch.from_numpy(data_p).double().to(device)
vot_reg = VotReg(data_p[:, 1:], data_e[:, 1:], data_p[:, 0], data_e[:, 0], device=device, verbose=False)
print("running regularized Wasserstein clustering...")
tick = time.time()
_, pred_label_e = vot_reg.cluster(reg_type=1, reg=0.1, max_iter_h=3000, max_iter_p=5)  # 0: w/o regularization
tock = time.time()
print("total running time : {} seconds".format(tock-tick))

# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
print("[optional] distribute centroids into target domain...")
vot_reg.cluster(0, max_iter_p=1)

# ------- plot map ------ #
cp = np.array([utils.COLOR_BLUE, utils.COLOR_RED])[vot_reg.label_p.int().cpu().numpy(), :]
fig235 = plt.subplot(235)
utils.plot_otmap(vot_reg.data_p_original, vot_reg.data_p.detach(), fig235, color=cp, title='w/ reg map')

# ------ plot after ----- #
plt.subplot(236)
ce = np.array([utils.COLOR_LIGHT_BLUE, utils.COLOR_LIGHT_RED])[pred_label_e.int(), :]
cp = np.array([utils.COLOR_DARK_BLUE, utils.COLOR_RED])[vot_reg.label_p.int(), :]
utils.plot_otsamples(vot_reg.data_p.detach(), vot_reg.data_e,
                     color_p=cp, color_e=ce, title='w/ reg after')

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("rwm_potential.png")
plt.show()
