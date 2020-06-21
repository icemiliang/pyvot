# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VOT
import utils

np.random.seed(19)

# Generate data
mean1 = [0., -0.2]
cov1 = [[0.04, 0], [0, 0.04]]
x11, x12 = np.random.multivariate_normal(mean1, cov1, 500).T
x1 = np.stack((x11, x12), axis=1).clip(-0.99, 0.99)

mean2 = [0.5, 0.5]
cov2 = [[0.01, 0], [0, 0.01]]
x21, x22 = np.random.multivariate_normal(mean2, cov2, 200).T
x2 = np.stack((x21, x22), axis=1).clip(-0.99, 0.99)

mean3 = [-0.5, 0.5]
cov3 = [[0.01, 0], [0, 0.01]]
x31, x32 = np.random.multivariate_normal(mean3, cov3, 200).T
x3 = np.stack((x31, x32), axis=1).clip(-0.99, 0.99)

x = np.concatenate((x1, x2, x3), axis=0)

mean = [0.0, 0.0]
cov = [[0.02, 0], [0, 0.02]]
K = 3
y1, y2 = np.random.multivariate_normal(mean, cov, K).T
y = np.stack((y1, y2), axis=1).clip(-0.99, 0.99)

xmin, xmax, ymin, ymax = -.7, .8, -.65, .8


# ---------------kmeans---------------

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init=y).fit(x)

label = kmeans.predict(x)
y = kmeans.cluster_centers_

color_map = np.array([[237, 125, 49, 255], [112, 173, 71, 255], [91, 155, 213, 255]]) / 255

fig = plt.figure(figsize=(4, 4))
for i in range(1):
    ce = color_map[label]
    utils.scatter_otsamples(y, x, size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("kmeans.svg", bbox_inches='tight')
plt.savefig("kmeans.png", dpi=300, bbox_inches='tight')
plt.close(fig)


# --------------- OT ---------------
y_copy = y.copy()
x_copy = x.copy()

vwb = VOT(y_copy, [x_copy], verbose=False)
output = vwb.cluster(lr=0.5, max_iter_h=20, max_iter_y=1, beta=0.5, keep_idx=True)
idxs = output['idxs'][0]
idx = vwb.idx

for i in range(0, min(21, len(idxs))):
    fig = plt.figure(figsize=(4, 4))
    ce = color_map[idxs[i]]
    utils.scatter_otsamples(vwb.y, vwb.x[0], nop=True, size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("vwb_" + str(i) + ".svg", bbox_inches='tight')
    plt.savefig("vwb_" + str(i) + ".png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(4, 4))
for i in range(1):
    utils.scatter_otsamples(vwb.y_original, vwb.x[i])
plt.axis('off')
# plt.savefig("4_4/initial.svg", bbox_inches='tight')
plt.savefig("initial.png", dpi=300, bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(4, 4))
ce = color_map[idx[0]]
utils.scatter_otsamples(vwb.y, vwb.x[0], size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("4_4/vot.svg", bbox_inches='tight')
plt.savefig("vot.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# --------------- Unbalanced OT ---------------
y_copy = y.copy()
x_copy = x.copy()

vwb = VOT(y_copy, [x_copy], verbose=False)
output = vwb.cluster(lr=0.5, max_iter_h=20, max_iter_y=1, beta=0.5, keep_idx=True)
idxs = output['idxs'][0]
idx = vwb.idx

for i in range(0, min(21, len(idxs))):
    fig = plt.figure(figsize=(4, 4))
    ce = color_map[idxs[i]]
    utils.scatter_otsamples(vwb.y, vwb.x[0], nop=True, size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("uvwb_" + str(i) + ".svg", bbox_inches='tight')
    plt.savefig("uvwb_" + str(i) + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)

fig = plt.figure(figsize=(4, 4))
ce = color_map[idx[0]]
utils.scatter_otsamples(vwb.y, vwb.x[0], size_p=30, marker_p='o', color_x=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("uvwb.svg", bbox_inches='tight')
plt.savefig("uvwb.png", dpi=300, bbox_inches='tight')
