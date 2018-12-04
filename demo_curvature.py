# Regularized Wasserstein Means (RWM)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Dec 1st 2018

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

from mpl_toolkits.mplot3d import Axes3D

# ----- set up WM ------ #
ot = Vot()
ot.import_data_file(pfilename='data/vent/skel.csv', efilename='data/vent/m5.csv', label=False)
ot.setup(max_iter_p=5, max_iter_h=1500)

fig = plt.figure()
# ------- run WM -------- #
ot.cluster(reg='curvature')


ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ot.p_coor[:,0], ot.p_coor[:,1], ot.p_coor[:,2], marker='o', linewidth=2)
ax.scatter(ot.e_coor[0::10,0], ot.e_coor[0::10,1], ot.e_coor[0::10,2], marker='o', s=0.2)

for i in range(22):
    ax.text(ot.p_coor[i,0], ot.p_coor[i,1], ot.p_coor[i,2], str(i))


plt.show()
