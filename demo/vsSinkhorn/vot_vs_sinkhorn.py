# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 25th 2020
# Licence: MIT

import os
import sys
import time
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import ot
import ot.plot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VOT
import utils


# -------------------- #
# ----- SET UP ------- #
# -------------------- #
mean, cov = [0, 0], [[.08, 0], [0, .08]]
# larger N -> faster convergence
N0, K = 100, 100
np.random.seed(0)
y = np.random.multivariate_normal(mean, cov, K).clip(-0.99, 0.99)
p_coor_before = y.copy()
x, _ = utils.random_sample(N0, 2, sampling='disk')
plt.close('all')
plt.figure(figsize=(8, 8))
minx, maxx, miny, maxy = -1, 1, -1, 1
plot_map = True


# -------------------- #
# ------- VOT -------- #
# -------------------- #
print('---- Running VOT ----')
dist = cdist(y, x, 'sqeuclidean')

mass_e = np.ones(N0) / N0
mass_p = np.ones(K) / K

tick = time.clock()
vot = VOT(y=y, x=x, verbose=False)
output = vot.cluster(max_iter_y=1, max_iter_h=3000, lr=1, lr_decay=200, beta=0.9)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))


if plot_map:
    # ----- plot before ----- #
    plt.subplot(331)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('before')
    plt.scatter(vot.x[0][:, 0], vot.x[0][:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # ------ plot map ------- #
    ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), vot.y.tolist())]
    lines = mc.LineCollection(ot_map, colors=utils.COLOR_LIGHT_GREY)
    fig332 = plt.subplot(332)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('VOT map ({0:.4f} s)'.format(tock - tick))
    fig332.add_collection(lines)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE, zorder=2)
    plt.scatter(vot.y[:, 0], vot.y[:, 1], marker='x', color=utils.COLOR_RED, zorder=3)

    # ------ plot after ----- #
    plt.subplot(333)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.grid(True)
    plt.title('VOT after')
    plt.scatter(vot.x[0][:, 0], vot.x[0][:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(vot.y[:, 0], vot.y[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)


# -------------------- #
# -- LINEAR PROGRAM -- #
# -------------------- #
print('---- Running linear program ----')

y = p_coor_before.copy()
M = ot.dist(y, x)
M /= M.max()

tick = time.clock()
Gs = ot.emd(mass_p, mass_e, M)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
for i in range(2):
    tmp = x[:, i]
    y[:, i] = (Gs * N0 * tmp[None, :]).sum(axis=1)


if plot_map:
    # ------ plot map ------- #
    plt.subplot(335); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('LP OT map ({0:.4f} s)'.format(tock-tick))
    ot.plot.plot2D_samples_mat(p_coor_before, x, Gs, color=[.5, .5, 0.5])
    plt.scatter(x[:, 0], x[:, 1], marker='x', color=utils.COLOR_RED, zorder=2)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE, zorder=3)

    # ------ plot after ----- #
    plt.subplot(336); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('LP OT after')
    plt.scatter(x[:, 0], x[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(y[:, 0], y[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)


# -------------------- #
# ----- SINKHORN ----- #
# -------------------- #
print('---- Running Sinkhorn ----')

y = p_coor_before.copy()
M = ot.dist(y, x)
M /= M.max()

lambd = 1e-3
tick = time.clock()
Gs = ot.sinkhorn(mass_p, mass_e, M, lambd)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
for i in range(2):
    tmp = x[:, i]
    y[:, i] = (Gs * N0 * tmp[None, :]).sum(axis=1)


if plot_map:
    # ------ plot map ------- #
    plt.subplot(338); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('Sinkhorn OT map ({0:.4f} s)'.format(tock-tick))
    ot.plot.plot2D_samples_mat(p_coor_before, x, Gs, color=[.5, .5, 0.5])
    plt.scatter(x[:, 0], x[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # ------ plot after ----- #
    plt.subplot(339); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
    plt.title('Sinkhorn OT after')
    plt.scatter(x[:, 0], x[:, 1], marker='x', color=utils.COLOR_RED)
    plt.scatter(y[:, 0], y[:, 1], marker='+', color=utils.COLOR_DARK_BLUE)

    # -------------------- #
    # ------- PLOT ------- #
    # -------------------- #
    plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
    plt.savefig('vot_vs_sinkhorn.png', format="png")
    plt.show()
