# Area Preserving via Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: July 6th 2019


"""
===============================================================
       Area Preserving Map through Optimal Transportation
===============================================================

This demo shows R^n -> R^n area preserving mapping through optimal transportation.
The total area is assumed to be one. We randomly sample a square and
count the samples to approximate the area. In this way, we avoid computing
convex hulls.

For now, PyVot assumes that the range in each dimension is (-1,1).
"""
from __future__ import print_function
from __future__ import division
# import non-vot stuffs
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VotAP
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import utils
import time
import numpy as np
from scipy.spatial import Delaunay


# ----- set up vot ------ #
mean = [0, 0]
cov = [[.08, 0], [0, .08]]
N = 50
data = np.random.multivariate_normal(mean, cov, N).clip(-0.99, 0.99)
ot = VotAP(data, sampling='square', ratio=100, verbose=True)

# ----- map ------ #
tick = time.time()
# vot.map(plot_filename='area_preserve.gif', max_iter=300)
e_idx, _ = ot.map(max_iter=3000)
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))
# TODO Area preserving usually requires a pre-defined boundary.
#  That is beyond the scope of the demo. Missing the boundary condition,
#  this area-preserving demo might not produce accurate maps near the boundary.
#  This can be visualized by drawing the Voronoi diagram or Delaunay triangulation
#  and one may see slight intersection near the boundary centroids.

# ----- plot before ----- #
X_p_before = np.copy(ot.data_p_original)
plt.figure(figsize=(12, 3.5))
plt.subplot(131); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('before')
plt.scatter(X_p_before[:, 0], X_p_before[:, 1], marker='o', color=utils.color_red, zorder=3)
tri = Delaunay(X_p_before)

# ------ plot map ------- #
X_p_after = np.copy(ot.data_p)
ot_map = [[tuple(p1), tuple(p2)] for p1, p2 in zip(X_p_before.tolist(), X_p_after.tolist())]
lines = mc.LineCollection(ot_map, colors=utils.color_light_grey)
fig232 = plt.subplot(132); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('area preserving map')
fig232.add_collection(lines)
plt.scatter(X_p_before[:, 0], X_p_before[:, 1], marker='o', color=utils.color_light_red, zorder=3)
plt.scatter(X_p_after[:, 0], X_p_after[:, 1], marker='o', facecolors='none', linewidth=2, color=utils.color_red, zorder=2)

# ------ plot after ----- #
plt.subplot(133); plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('after')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2, s=0.5)
color = plt.get_cmap('viridis')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], s=1, marker='o', color=color(e_idx / (N - 1)))
plt.scatter(X_p_after[:, 0], X_p_after[:, 1], marker='o', facecolors='none', linewidth=2, color=utils.color_red, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("ot_area_preserve.png")
plt.show()
# TODO plot Voronoi diagram
