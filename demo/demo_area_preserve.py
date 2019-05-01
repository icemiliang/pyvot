# Area Preserving via Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Jan 18th 2019

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

from vot import *
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import utils as vot
import time

# ----- set up ot ------ #
ot = VotAreaPreserve(max_iter=5000, ratio=200, rate=0.2, dim=2)
ot.import_data_from_file('data/p.csv', has_label=True)

# ----- plot before ----- #
p_coor_before = np.copy(ot.Xp)
plt.figure(figsize=(12,4))

cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(131); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('before')
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ----- run area preserving mapping ------ #
tick = time.clock()
ot.area_preserve()
tock = time.clock()
print(tock-tick)

# ------ plot map ------- #
p_coor_after = np.copy(ot.Xp)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig232 = plt.subplot(132); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('area preserving map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(133); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('after')
plt.scatter(ot.X_e[:, 0], ot.X_e[:, 1], marker='.', color=ce, zorder=2, s=0.5)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("ot_area_preserve.png")
plt.show()
