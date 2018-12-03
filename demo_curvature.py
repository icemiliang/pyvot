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
from mpl_toolkits import mplot3d

# ----- set up WM ------ #
ot = Vot()
ot.import_data_file(pfilename='data/vent/skel.csv', efilename='data/vent/m5.csv', label=False)
ot.setup(max_iter_p=1, max_iter_h=1500)

# ------- run WM -------- #
ot.cluster(reg=0)

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.axis('equal')
ax.scatter3D(ot.p_coor[:,0], ot.p_coor[:,1], ot.p_coor[:,2], marker='o', facecolors='none', linewidth=2, zorder=3)
plt.show()
