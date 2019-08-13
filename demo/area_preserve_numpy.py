# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Aug 11th 2019
# Licence: MIT

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

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VotAP
import utils


np.random.seed(0)
mean = [0, 0]
cov = [[.08, 0], [0, .08]]
N = 50

data_backup = np.random.multivariate_normal(mean, cov, N).clip(-0.99, 0.99)

# ----------------------------------- #
# ------------ Example 1 ------------ #
# ----------------------------------- #

# ----- set up vot ------ #
data = data_backup.copy()
vot = VotAP(data, sampling='square', ratio=200, verbose=True)

# ----- map ------ #
tick = time.time()
# vot.map(plot_filename='area_preserve.gif', max_iter=300)
e_idx, _ = vot.map(max_iter=3000)
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))

# Note: Area preserving usually requires a pre-defined boundary.
#  That is beyond the scope of the demo. Missing the boundary condition,
#  this area-preserving demo might not produce accurate maps near the boundary.
#  This can be visualized by drawing the Voronoi diagram or Delaunay triangulation
#  and one may see slight intersection near the boundary centroids.

# ----- plot before ----- #
plt.figure(figsize=(12, 8))
plt.subplot(231)
utils.plot_otsamples(vot.data_p_original, vot.data_e, title='before', )

# ------ plot map ------- #
fig232 = plt.subplot(232)
utils.plot_otmap(vot.data_p_original, vot.data_p, fig232, title='vot map', facecolor_after='none')

# ------ plot after ----- #
ce = np.array(plt.get_cmap('viridis')(e_idx / (N - 1)))
plt.subplot(233)
utils.plot_otsamples(vot.data_p, vot.data_e, color_e=ce, size_e=5, title='after', facecolor_p='none')


# ----------------------------------- #
# ------------ Example 2 ------------ #
# ----------------------------------- #

# ----- set up vot ------ #
data = data_backup.copy()
vot2 = VotAP(data, sampling='circle', ratio=200, verbose=True)

# ----- map ------ #
tick = time.time()
# vot.map(plot_filename='area_preserve.gif', max_iter=300)
e_idx, _ = vot2.map(max_iter=3000)
tock = time.time()
print('total time: {0:.4f}'.format(tock-tick))

# ----- plot before ----- #
plt.subplot(234)
utils.plot_otsamples(vot2.data_p_original, vot2.data_e, title='before')

# ------ plot map ------- #
fig235 = plt.subplot(235)
utils.plot_otmap(vot2.data_p_original, vot2.data_p, fig235, title='vot map', facecolor_after='none')

# ------ plot after ----- #
ce = np.array(plt.get_cmap('viridis')(e_idx / (N - 1)))
plt.subplot(236)
utils.plot_otsamples(vot2.data_p, vot2.data_e, size_p=30, color_e=ce, size_e=5, title='after', facecolor_p='none')

# ---- plot and save ---- #
plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.5)
# plt.savefig("vot_area_preserve_numpy.png")
plt.show()
