# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_torch import VWB, UVWB
import utils

np.random.seed(19)

# Generate data
mean1 = [0., -0.2]
cov1 = [[0.04, 0], [0, 0.04]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
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


# ---------------kmeans---------------

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init=x).fit(x0)

label = kmeans.predict(x0)
newx = kmeans.cluster_centers_

color_map = np.array([[237, 125, 49, 255], [112, 173, 71, 255], [91, 155, 213, 255]]) / 255

plt.figure(figsize=(4, 4))
for i in range(1):
    ce = color_map[label]
    utils.scatter_otsamples(newx, x0, size_p=30, marker_p='o', color_e=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("kmeans.svg", bbox_inches='tight')
plt.savefig("kmeans.png", dpi=300, bbox_inches='tight')

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# ---------------VWB---------------
x = newx
x_copy = torch.from_numpy(x)
x0_copy = torch.from_numpy(x0)

vwb = VWB(x_copy, [x0_copy], device=device, verbose=False)
output = vwb.cluster(lr=0.5, max_iter_h=1000, max_iter_p=1, beta=0.5)

idx, e_idxss = output['idx'], output['idxs']

idxss_cat = torch.stack(e_idxss[0])
idxss_cat = idxss_cat.numpy()

for i in range(0, min(21, len(idxss_cat))):
    fig = plt.figure(figsize=(4, 4))
    ce = color_map[idxss_cat[i]]
    utils.scatter_otsamples(vwb.data_p, vwb.data_e[0], nop=True, size_p=30, marker_p='o', color_e=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("vwb_" + str(i) + ".svg", bbox_inches='tight')
    plt.savefig("vwb_" + str(i) + ".png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(4, 4))
for i in range(1):
    utils.scatter_otsamples(vwb.data_p_original, vwb.data_e[i])
plt.axis('off')
# plt.savefig("4_4/initial.svg", bbox_inches='tight')
plt.savefig("initial.png", dpi=300, bbox_inches='tight')

fig = plt.figure(figsize=(4, 4))
ce = color_map[idx[0]]
utils.scatter_otsamples(vwb.data_p, vwb.data_e[0], size_p=30, marker_p='o', color_e=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("4_4/vot.svg", bbox_inches='tight')
plt.savefig("vot.png", dpi=300, bbox_inches='tight')

# ---------------UVWB---------------
x_copy = torch.from_numpy(x)
x0_copy = torch.from_numpy(x0)

vwb = UVWB(x_copy, [x0_copy], device=device, verbose=False)
out = vwb.cluster(lr=0.5, max_iter_h=1000, max_iter_p=1, beta=0.5)

e_idx, e_idxss = output['idx'], output['idxs']

e_idxss_cat = torch.stack(e_idxss[0])
e_idxss_cat = e_idxss_cat.numpy()

for i in range(0, min(21, len(e_idxss_cat))):
    fig = plt.figure(figsize=(4, 4))
    ce = color_map[e_idxss_cat[i]]
    utils.scatter_otsamples(vwb.data_p, vwb.data_e[0], nop=True, size_p=30, marker_p='o', color_e=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("uvwb_" + str(i) + ".svg", bbox_inches='tight')
    plt.savefig("uvwb_" + str(i) + ".png", dpi=300, bbox_inches='tight')

fig = plt.figure(figsize=(4, 4))
ce = color_map[e_idx[0]]
utils.scatter_otsamples(vwb.data_p, vwb.data_e[0], size_p=30, marker_p='o', color_e=ce, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("uvwb.svg", bbox_inches='tight')
plt.savefig("uvwb.png", dpi=300, bbox_inches='tight')
