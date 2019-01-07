# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Jan 7th 2019

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
import sklearn.datasets

# Generate data
Ne = 5000
Np = 100
Xe, ye = sklearn.datasets.make_moons(n_samples=Ne, noise=0.05, random_state=1)
Xp, yp = sklearn.datasets.make_moons(n_samples=Np, noise=0.05, random_state=1)
Xp[:,0] -= 0.5; Xp[:,1] -= 0.25
Xe[:,0] -= 0.5; Xe[:,1] -= 0.25

theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c,-s), (s, c)))
Xp = Xp.dot(R)
yp = yp.transpose()

Xp1 = Xp.copy()
Xe1 = Xe.copy()

# -------------------------------------- #
# --------- w/o regularization --------- #
# -------------------------------------- #

# ----- set up WM ------ #
ot = Vot()
# ot.import_data_file('data/p.csv','data/e.csv')
ot.import_data(Xp, Xe, yp = yp, ye = ye)
ot.setup(max_iter_p = 5, max_iter_h = 2000)

# ----- plot before ----- #
p_coor_before = np.copy(ot.p_coor)
plt.figure(figsize=(12,8))
xmin = -2.0; xmax = 2.0; ymin = -1.5; ymax = 1.5

cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(231); plt.xlim(xmin,xmax); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/o reg before')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=vot.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ------- run WM -------- #
ot.cluster(0) # 0: w/o regularization

# ------ plot map ------- #
p_coor_after = np.copy(ot.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig232 = plt.subplot(232); plt.xlim(xmin,xmax); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/o reg map')
fig232.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot.e_predict)
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(233); plt.xlim(xmin,xmax); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/o reg after')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# -------------------------------------- #
# --------- w/ regularization ---------- #
# -------------------------------------- #



ot_reg = Vot()
# ot.import_data_file('data/p.csv','data/e.csv')
ot_reg.import_data(Xp1, Xe1, yp = yp, ye = ye)
ot_reg.setup(max_iter_p = 10, max_iter_h = 2000)

# ----- plot before ----- #
cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot_reg.p_label)]
plt.subplot(234); plt.xlim(xmin,1.5); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/ reg before')
plt.scatter(ot_reg.e_coor[:,0], ot_reg.e_coor[:,1], marker='.', color=vot.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ------- run RWM ------- #
ot_reg.cluster(reg_type = 'transform', reg = 50.0)

# ------- run OT ------- #
# Compute OT one more time to disperse the centroids into the empirical domain.
# This almost does not change the correspondence but can give better positions.
# This is optional.
ot_reg.setup(max_iter_p = 1, max_iter_h = 2000)
ot_reg.cluster()

# ------- plot map ------ #
p_coor_after = np.copy(ot_reg.p_coor)
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(), p_coor_after.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig235 = plt.subplot(235); plt.xlim(xmin,xmax); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/ reg map')
fig235.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ------ plot after ----- #
le = np.copy(ot_reg.e_predict)
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(le)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot_reg.p_label)]
plt.subplot(236); plt.xlim(xmin,xmax); plt.ylim(ymin,ymax); plt.grid(True); plt.title('w/ reg after')
plt.scatter(ot_reg.e_coor[:,0], ot_reg.e_coor[:,1], marker='.', color=ce, zorder=2)
plt.scatter(p_coor_after[:,0], p_coor_after[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
plt.savefig("data/rwm_transform.png")
plt.show()
