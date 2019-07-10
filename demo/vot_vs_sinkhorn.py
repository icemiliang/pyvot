# Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: July 6th 2019

from __future__ import print_function
from __future__ import division
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from vot import Vot
import utils
import ot
import ot.plot
from scipy.spatial.distance import cdist


# -------------------- #
# ----- SET UP ------- #
# -------------------- #
mean, cov = [0, 0], [[.08, 0], [0, .08]]
N, M = 100, 100
data_p = np.random.multivariate_normal(mean, cov, N).clip(-0.99, 0.99)
p_coor_before = data_p.copy()
data_e, _ = utils.random_sample(M, 2, sampling='disk')
plt.close('all')
plt.figure(figsize=(12, 8))
minx, maxx, miny, maxy = -1, 1, -1, 1


# -------------------- #
# ------- VOT -------- #
# -------------------- #
print('---- Running VOT ----')
dist = cdist(data_p, data_e) ** 2
tick = time.clock()
vot = Vot(data_p=data_p, data_e=data_e, verbose=False)
vot.update_map(dist, max_iter=5000, lr=0.3)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
vot.update_p()

# ----- plot before ----- #
plt.subplot(231); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True); plt.title('before')
plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='x', color=utils.color_red)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.color_dark_blue)

# ------ plot map ------- #
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(p_coor_before.tolist(), vot.data_p.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True);
plt.title('VOT map ({0:.4f} s)'.format(tock-tick))
fig232.add_collection(lines)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.color_dark_blue, zorder=2)
plt.scatter(vot.data_p[:, 0], vot.data_p[:, 1], marker='x', color=utils.color_red, zorder=3)

# ------ plot after ----- #
plt.subplot(233); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True); plt.title('VOT after')
plt.scatter(vot.data_e[:, 0], vot.data_e[:, 1], marker='x', color=utils.color_red)
plt.scatter(vot.data_p[:, 0], vot.data_p[:, 1], marker='+', color=utils.color_dark_blue)

# -------------------- #
# ----- SINKHORN ----- #
# -------------------- #
print('---- Running Sinkhorn ----')
a, b = np.ones((N,)) / N, np.ones((M,)) / M

data_p = p_coor_before.copy()
M = ot.dist(data_p, data_e)
M /= M.max()

lambd = 1e-3
tick = time.clock()
Gs = ot.sinkhorn(a, b, M, lambd)
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
for i in range(2):
    tmp = data_e[:, i]
    data_p[:, i] = (Gs * N * tmp[None, :]).sum(axis=1)

# ------ plot map ------- #
plt.subplot(235); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
plt.title('Sinkhorn OT map ({0:.4f} s)'.format(tock-tick))
ot.plot.plot2D_samples_mat(p_coor_before, data_e, Gs, color=[.5, .5, 0.5])
plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.color_red)
plt.scatter(p_coor_before[:, 0], p_coor_before[:, 1], marker='+', color=utils.color_dark_blue)

# ------ plot after ----- #
plt.subplot(236); plt.xlim(minx, maxx); plt.ylim(miny, maxy); plt.grid(True)
plt.title('Sinkhorn OT after')
plt.scatter(data_e[:, 0], data_e[:, 1], marker='x', color=utils.color_red)
plt.scatter(data_p[:, 0], data_p[:, 1], marker='+', color=utils.color_dark_blue)

# -------------------- #
# ------- PLOT ------- #
# -------------------- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
plt.savefig('vot_vs_sinkhorn.png', format="png")
plt.show()

pass
