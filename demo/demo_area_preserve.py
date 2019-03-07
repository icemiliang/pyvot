# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: MArch 6th 2019

"""
===============================================================
       Area Preserving Map through Optimal Transportation
===============================================================

This demo shows area preserving mapping through optimal transportation.
The total area is assumed to be one. We randomly sample a square and
count the samples to approximate the area. In this way, we avoid computing
convex hulls.

For now, PyVot assumes that the range in each dimension is (-1,1).
"""
from __future__ import print_function
from __future__ import division
# import non-vot stuffs
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
# import vot stuffs
import vot
import utils

# ----- set up ot ------ #
ot = vot.VotAreaPreserve()
ot.import_data_from_file('data/p.csv')
ot.setup(max_iter=5000, ratio=200, rate=0.2, dim = 2)

# ----- plot before ----- #
p_coor_before = np.copy(ot.p_coor)
plt.figure(figsize=(12,8))

cp = [utils.color_blue, utils.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(231); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('before')
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ----- run area preserving mapping ------ #
print("running area-preserving mapping...")
tick = time.clock()
ot.area_preserve() # 0: w/o regularization
tock = time.clock()
print( "running time: %.2f seconds" % (tock-tick))

# ------ plot map ------- #
p_coor_after = np.copy(ot.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=utils.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('area preserving map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(233); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('after')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2, s=0.5)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ----- set up ot ------ #
ot = vot.VotAreaPreserve()
ot.import_data_from_file('data/p_random_weight.csv', mass=True)
ot.setup(max_iter=5000, ratio=200, rate=0.2, dim = 2)

# ----- plot before ----- #
p_coor_before = np.copy(ot.p_coor)

cp = [utils.color_blue, utils.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(234); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('before')
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ----- run area preserving mapping ------ #
print("running area-preserving mapping...")
tick = time.clock()
ot.area_preserve() # 0: w/o regularization
tock = time.clock()
print( "running time: %.2f seconds" % (tock-tick))

# ------ plot map ------- #
p_coor_after = np.copy(ot.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=utils.color_light_grey)
fig232 = plt.subplot(235); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('area preserving map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [utils.color_light_blue, utils.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [utils.color_dark_blue, utils.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(236); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('after')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2, s=0.5)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("ot_area_preserve.png")
# plt.show()

# plt.figure()
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(p_coor_after)
import matplotlib.pyplot as plt
voronoi_plot_2d(vor)
plt.show()
