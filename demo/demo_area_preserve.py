# Area Preserving via Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 2nd 2019

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

from vot import VotAP
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import utils
import time
import numpy as np
from scipy.spatial import Delaunay


# ----- set up ot ------ #
ot = VotAP(ratio=500)
# data = np.loadtxt('data/p_ap.csv', delimiter=",")
mean = [0, 0]
cov = [[.08, 0],[0, .04]]
data = np.random.multivariate_normal(mean, cov, 50).clip(-0.99,0.99)
ot.import_data(data)

tick = time.clock()
ot.area_preserve(sampling='unisquare')
tock = time.clock()
print('total time: {0:.4f}'.format(tock-tick))
# TODO Area preserving requires a consistent boundary, (i.e.) That is a todo.
#  That is also why triangles may intersect with each other near the boundary.

# ----- plot before ----- #
X_p_before = np.copy(ot.X_p_original)
plt.figure(figsize=(12,4))
plt.subplot(131); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('before')
plt.scatter(X_p_before[:, 0], X_p_before[:, 1], marker='o', color=utils.color_red, zorder=3)
tri = Delaunay(X_p_before)
plt.triplot(X_p_before[:, 0], X_p_before[:, 1], tri.simplices)

# ------ plot map ------- #
X_p_after = np.copy(ot.X_p)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(X_p_before.tolist(), X_p_after.tolist())]
lines = mc.LineCollection(map, colors=utils.color_light_grey)
fig232 = plt.subplot(132); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('area preserving map')
fig232.add_collection(lines)
plt.scatter(X_p_before[:, 0], X_p_before[:, 1], marker='o', color=utils.color_light_red, zorder=3)
plt.scatter(X_p_after[:, 0], X_p_after[:, 1], marker='o', facecolors='none', linewidth=2, color=utils.color_red, zorder=2)

# ------ plot after ----- #
plt.subplot(133); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('after')
plt.scatter(ot.X_e[:, 0], ot.X_e[:, 1], marker='.', color=utils.color_light_grey, zorder=2, s=0.5)
plt.scatter(X_p_after[:, 0], X_p_after[:, 1], marker='o', facecolors='none', linewidth=2, color=utils.color_red, zorder=3)
plt.triplot(X_p_after[:, 0], X_p_after[:, 1], tri.simplices)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("ot_area_preserve.png")
plt.show()
