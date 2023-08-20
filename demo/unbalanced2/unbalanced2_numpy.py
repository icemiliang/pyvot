# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_numpy import VOT
import utils_numpy as utils

np.random.seed(19)

# Generate data
mean1 = [-0.5, 0.5]
cov1 = [[0.03, 0], [0, 0.01]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 5000).T
x1 = np.stack((x1, y1), axis=1).clip(-0.99, 0.99)

mean2 = [0.5, 0.5]
cov2 = [[0.01, 0.], [0., 0.03]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
x2 = np.stack((x2, y2), axis=1).clip(-0.99, 0.99)

mean = [0.0, -0.5]
cov = [[0.02, 0], [0, 0.02]]
K = 50
x, y = np.random.multivariate_normal(mean, cov, K).T
x = np.stack((x, y), axis=1).clip(-0.99, 0.99)


vot = VOT(x, [x1, x2], verbose=False)
vot.cluster(max_iter_h=3000, max_iter_y=1)
idx = vot.idx

xmin, xmax, ymin, ymax = -1., 1., 0., 1.


for k in [21]:
    plt.figure(figsize=(8, 4))
    for i in range(2):
        ce = np.array(plt.get_cmap('viridis')(idx[i] / (K - 1)))
        utils.scatter_otsamples(vot.y, vot.x[i], size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')

    p = vot.y[k]

    for i in range(2):
        es = vot.x[i][idx[i] == k]
        for e in es:
            x = [p[0], e[0]]
            y = [p[1], e[1]]
            plt.plot(x, y, c='lightgray', alpha=0.4)

    # plt.savefig("ship" + str(k) + ".svg")
    plt.savefig("ship" + str(k) + ".png", dpi=300, bbox_inches='tight')
