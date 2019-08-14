# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Aug 11th 2019
# Licence: MIT

import os
import sys
import time
import ot
import ot.plot
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import Vot
import utils


# -------------------- #
# ----- SET UP ------- #
# -------------------- #
mean, cov = [0, 0], [[.08, 0], [0, .08]]
# larger N -> faster convergence
N, K = 1000, 100
np.random.seed(0)
data_p = np.random.multivariate_normal(mean, cov, K).clip(-0.99, 0.99)
p_coor_before = data_p.copy()
data_e, _ = utils.random_sample(N, 2, sampling='disk')
plt.close('all')
plt.figure(figsize=(8, 8))
minx, maxx, miny, maxy = -1, 1, -1, 1
plot_map = True


# -------------------- #
# ------- VOT -------- #
# -------------------- #
print('---- Running VOT ----')
dist = cdist(data_p, data_e) ** 2

mass_e = np.ones(N) / N
mass_p = np.ones(K) / K

tick = time.clock()
vot = Vot(data_p=data_p, data_e=data_e, weight_e=mass_e, weight_p=mass_p, verbose=False)
e_idx, _ = vot.update_map(dist, max_iter=5000, lr=0.5, lr_decay=200, beta=0.9)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
vot.update_p(e_idx)

if plot_map:
    # ----- plot before ----- #
    plt.subplot(331)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('before')
    plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # ------ plot map ------- #
    ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), vot.data_p.tolist())]
    lines = mc.LineCollection(ot_map, colors=utils.COLOR_LIGHT_GREY)
    fig332 = plt.subplot(332)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('VOT map ({0:.4f} s)'.format(tock - tick))
    fig332.add_collection(lines)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE, zorder=2)
    plt.scatter(vot.data_p[:, 0], vot.data_p[:, 1], marker='x', color=utils.COLOR_RED, zorder=3)

    # ------ plot after ----- #
    plt.subplot(333)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('VOT after')
    plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(vot.data_p[:, 0], vot.data_p[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)


# -------------------- #
# -- LINEAR PROGRAM -- #
# -------------------- #
print('---- Running linear program ----')

data_p = p_coor_before.copy()
M = ot.dist(data_p, data_e)
M /= M.max()

tick = time.clock()
Gs = ot.emd(mass_p, mass_e, M)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
for i in range(2):
    tmp = data_e[:, i]
    data_p[:, i] = (Gs * N * tmp[None, :]).sum(axis=1)


if plot_map:
    # ------ plot map ------- #
    plt.subplot(335); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('LP OT map ({0:.4f} s)'.format(tock-tick))
    ot.plot.plot2D_samples_mat(p_coor_before, data_e, Gs, color=[.5, .5, 0.5])
    plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.COLOR_RED, zorder=2)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE, zorder=3)

    # ------ plot after ----- #
    plt.subplot(336); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('LP OT after')
    plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(data_p[:, 0], data_p[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)


# -------------------- #
# ----- SINKHORN ----- #
# -------------------- #
print('---- Running Sinkhorn ----')

data_p = p_coor_before.copy()
M = ot.dist(data_p, data_e)
M /= M.max()

lambd = 1e-3
tick = time.clock()
Gs = ot.sinkhorn(mass_p, mass_e, M, lambd)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
for i in range(2):
    tmp = data_e[:, i]
    data_p[:, i] = (Gs * N * tmp[None, :]).sum(axis=1)


if plot_map:
    # ------ plot map ------- #
    plt.subplot(338); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('Sinkhorn OT map ({0:.4f} s)'.format(tock-tick))
    ot.plot.plot2D_samples_mat(p_coor_before, data_e, Gs, color=[.5, .5, 0.5])
    plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # ------ plot after ----- #
    plt.subplot(339); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('Sinkhorn OT after')
    plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(data_p[:, 0], data_p[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # -------------------- #
    # ------- PLOT ------- #
    # -------------------- #
    plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
    # plt.savefig('vot_vs_sinkhorn.png', format="png")
    plt.show()
