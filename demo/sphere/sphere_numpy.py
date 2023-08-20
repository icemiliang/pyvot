# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_numpy import VOT


np.random.seed(19)

mean1 = [0.0, 0.0]
cov1 = [[0.3, 0], [0, 0.3]]
u1, v1 = np.random.multivariate_normal(mean1, cov1, 1000).T
u1 = u1 * np.pi / 8 + np.pi / 2
v1 = v1 * np.pi / 8 + np.pi * 1 / 4
x11 = np.cos(u1) * np.sin(v1)
x12 = np.sin(u1) * np.sin(v1)
x13 = np.cos(v1)

mean2 = [0.0, 0.0]
cov2 = [[0.4, 0], [0, 0.4]]
u2, v2 = np.random.multivariate_normal(mean2, cov2, 1000).T
u2 = u2 * np.pi / 8 + np.pi
v2 = v2 * np.pi / 8 + np.pi * 1 / 4
x21 = np.cos(u2) * np.sin(v2)
x22 = np.sin(u2) * np.sin(v2)
x23 = np.cos(v2)

mean0 = [0.0, 0.0]
cov0 = [[0.3, 0], [0, 0.3]]
K = 50
u0, v0 = np.random.multivariate_normal(mean0, cov0, K).T
u0 = u0 * np.pi / 8 + np.pi * 3 / 4
v0 = v0 * np.pi / 8 + np.pi * 1 / 4
y1 = np.cos(u0) * np.sin(v0)
y2 = np.sin(u0) * np.sin(v0)
y3 = np.cos(v0)


plt.show()

x1 = np.stack((x11, x12, x13), axis=1).clip(-0.99, 0.99)
x2 = np.stack((x21, x22, x23), axis=1).clip(-0.99, 0.99)
y = np.stack((y1, y2, y3), axis=1).clip(-0.99, 0.99)

y /= np.linalg.norm(y, axis=1, keepdims=True)
x1 /= np.linalg.norm(x1, axis=1, keepdims=True)
x2 /= np.linalg.norm(x2, axis=1, keepdims=True)

vwb = VOT(y, [x1, x2], verbose=False)
vwb.cluster(max_iter_h=5000, max_iter_y=1, space='spherical')
idx = vwb.idx

xmin, xmax, ymin, ymax = -1.0, 1.0, -0.5, 0.5
u, v = np.mgrid[np.pi/4:np.pi*5/4:1000j, np.pi/2:np.pi*3/2:1000j]
gx = np.cos(u)*np.sin(v)
gy = np.sin(u)*np.sin(v)
gz = np.cos(v)

for k in [12]:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.magma((gx - gx.min()) / float((gx - gx.min()).max()))
    ax.plot_surface(gx * 0.95, gy * 0.95, gz * 0.95, antialiased=False, facecolors=colors, linewidth=0, shade=False)
    for i in range(2):
        ce = np.array(plt.get_cmap('viridis')(idx[i] / (K - 1)))
        ax.scatter(vwb.x[i][:, 0], vwb.x[i][:, 1], vwb.x[i][:, 2], s=1, color=ce, zorder=2)
    ax.scatter(vwb.y[:, 0], vwb.y[:, 1], vwb.y[:, 2], s=5, marker='o',
               facecolors='none', linewidth=2, color='r', zorder=5)

    e0s = vwb.x[0][idx[0] == k]
    e1s = vwb.x[1][idx[1] == k]
    p = vwb.y[k]

    for e0, e1 in zip(e0s, e1s):
        x = [e1[0], p[0], e0[0]]
        y = [e1[1], p[1], e0[1]]
        z = [e1[2], p[2], e0[2]]
        plt.plot(x, y, z, c='gray', alpha=0.4, zorder=5)
    ax.view_init(elev=10., azim=100.)
    plt.axis('off')
    # plt.savefig("sphere_" + str(k) + ".svg", bbox_inches='tight')
    plt.savefig("sphere_" + str(k) + ".png", dpi=300, bbox_inches='tight')
