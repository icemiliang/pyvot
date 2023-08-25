# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_torch import VOT
import utils_torch as utils

np.random.seed(19)

# ------------ Generate data ------------- #
mean1 = [0., -0.2]
cov1 = [[0.05, 0], [0, 0.05]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
x1 = np.stack((x1, y1), axis=1).clip(-0.99, 0.99)

mean2 = [0.5, 0.5]
cov2 = [[0.01, 0], [0, 0.01]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 200).T
x2 = np.stack((x2, y2), axis=1).clip(-0.99, 0.99)

mean3 = [-0.5, 0.5]
cov3 = [[0.01, 0], [0, 0.01]]
x3, y3 = np.random.multivariate_normal(mean3, cov3, 200).T
x3 = np.stack((x3, y3), axis=1).clip(-0.99, 0.99)

x0 = np.concatenate((x1, x2, x3), axis=0)

mean = [0.0, 0.0]
cov = [[0.02, 0], [0, 0.02]]
K = 3
x, y = np.random.multivariate_normal(mean, cov, K).T
x = np.stack((x, y), axis=1).clip(-0.99, 0.99)

xmin, xmax, ymin, ymax = -.7, .8, -.65, .8

plt.close()
# ---------------kmeans---------------

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init=x).fit(x0)

label = kmeans.predict(x0)
newx = kmeans.cluster_centers_

color_map = np.array([[237, 125, 49, 255], [112, 173, 71, 255], [91, 155, 213, 255]]) / 255

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# ---------------VWB---------------
x = newx

for reg in [0.5, 2, 1e9]:

    x_copy = torch.from_numpy(x)
    x0_copy = torch.from_numpy(x0)

    vot = VOT(x_copy, [x0_copy], device=device, verbose=False)
    vot.cluster(lr=0.5, max_iter_h=1000, max_iter_y=1, beta=0.5, reg=reg)

    idx = vot.idx

    fig = plt.figure(figsize=(4, 4))
    ce = color_map[idx[0]]
    utils.scatter_otsamples(vot.y, vot.x[0], size_p=30, marker_p='o', color_x=ce,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("0.svg", bbox_inches='tight')
    plt.savefig(str(reg) + ".png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(4, 4))

ce = color_map[label]
utils.scatter_otsamples(newx, x0, size_p=30, marker_p='o', color_x=ce,
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("0.svg", bbox_inches='tight')
plt.savefig("0.png", dpi=300, bbox_inches='tight')
