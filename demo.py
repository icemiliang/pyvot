# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Dec 1st 2018

"""
===========================================
    Variational Wasserstein Clustering
===========================================

This demo shows how to use variational transportation to cluster
empirical samples into known clusters.

Predicted labels of the empirical samples come from the centroids.
It is equivalent to 1NN w.r.t. the power Euclidean distance.
"""

from vot import *
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import utils as vot

# ----- set up vwc ------ #
ot = Vot()
ot.import_data_file('data/p.csv','data/e.csv')
ot.setup(max_iter_p = 2, max_iter_h = 2500)

# ----- plot before ----- #
p_coor_before =  np.copy(ot.p_coor)
plt.figure(figsize=(12,3.5))

cp = [vot.color_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(131); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('Before')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=vot.color_light_grey, zorder=2)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp, zorder=3)

# ------- run vwc ------- #
ot.cluster()

# ------ plot map ------- #
map = [[tuple(p1),tuple(p2)] for p1,p2 in zip(p_coor_before.tolist(),ot.p_coor.tolist())]
lines = mc.LineCollection(map, colors=vot.color_light_grey)
fig132 = plt.subplot(132); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('Map')
fig132.add_collection(lines)
plt.scatter(p_coor_before[:,0], p_coor_before[:,1], marker='o', color=cp,zorder=3)
plt.scatter(ot.p_coor[:,0], ot.p_coor[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=2)

# ----- plot after ------ #
ce = [vot.color_light_blue, vot.color_light_red]
ce = [ce[label] for index,label in np.ndenumerate(ot.e_predict)]
cp = [vot.color_dark_blue, vot.color_red]
cp = [cp[label] for index,label in np.ndenumerate(ot.p_label)]
plt.subplot(133); plt.xlim(-1,1); plt.ylim(-1,1); plt.grid(True); plt.title('After')
plt.scatter(ot.e_coor[:,0], ot.e_coor[:,1], marker='.', color=ce, zorder=2)
plt.scatter(ot.p_coor[:,0], ot.p_coor[:,1], marker='o', facecolors='none', linewidth=2, color=cp, zorder=3)

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=1.0)
plt.axis('off')
# plt.savefig("vwc.png")
plt.show()
