# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Dec 2nd 2018

"""
===========================================
       Regularized Wasserstein Means
===========================================

This demo shows that regularizing the centroids by using class labels
and pairwise distances can benefit domain adaptation applications.

Predicted labels of the empirical samples come from the centroids.
It is equivalent to 1NN w.r.t. the power Euclidean distance.
"""

from vot import *
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import utils as vot

# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #

# ----- set up WM ------ #
ot = Vot()
ot.import_data_file('data/p.csv','data/e.csv')
ot.setup(max_iter_p=1, max_iter_h=1500)

# ----- plot before ----- #
p_coor_before = np.copy(ot.p_coor)
plt.figure(figsize=(12,8))

cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(231); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w/o reg before')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=vot.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ------- run WM -------- #
ot.cluster(0) # 0: w/o regularization

# ------ plot map ------- #
p_coor_after = np.copy(ot.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w/o reg map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(233); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w/o reg after')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #

ot = Vot()
ot.import_data_file('data/p.csv','data/e.csv')
ot.setup(max_iter_p=5, max_iter_h=1500)

# ----- plot before ----- #
cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(234); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w reg before')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=vot.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ------- run RWM ------- #
ot.cluster(reg = 'potential', alpha = 0.01)

# ------- plot map ------ #
p_coor_after = np.copy(ot.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig235 = plt.subplot(235); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w reg map')
fig235.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(236); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('w reg after')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
plt.savefig("data/rwm.png")
plt.show()
