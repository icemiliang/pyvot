import os
import sys
import torch
import numpy as np
import matplotlib
from mpl_toolkits import mplot3d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_pytorch import SVWB


np.random.seed(19)

u, v = np.mgrid[np.pi/4:np.pi*5/4:1000j, np.pi/2:np.pi*3/2:1000j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, color='gray', s=0.1)
# plt.show()


mean1 = [0.0, 0.0]
cov1 = [[0.3, 0], [0, 0.3]]
u1, v1 = np.random.multivariate_normal(mean1, cov1, 1000).T
u1 = u1 * np.pi / 8 + np.pi / 2
v1 = v1 * np.pi / 8 + np.pi * 1 / 4
x1 = np.cos(u1) * np.sin(v1)
y1 = np.sin(u1) * np.sin(v1)
z1 = np.cos(v1)
# ax.scatter(x1, y1, z1, color='b', s=0.5)

mean2 = [0.0, 0.0]
cov2 = [[0.4, 0], [0, 0.4]]
u2, v2 = np.random.multivariate_normal(mean2, cov2, 1000).T
u2 = u2 * np.pi / 8 + np.pi
v2 = v2 * np.pi / 8 + np.pi * 1 / 4
x2 = np.cos(u2) * np.sin(v2)
y2 = np.sin(u2) * np.sin(v2)
z2 = np.cos(v2)
# ax.scatter(x2, y2, z2, color='b', s=0.5)

# mean3 = [0.0, 0.0]
# cov3 = [[0.4, 0], [0, 0.4]]
# u3, v3 = np.random.multivariate_normal(mean3, cov3, 1000).T
# u3 = u3 * np.pi / 8 + np.pi * 3 / 4
# v3 = v3 * np.pi / 8 + np.pi * 1 / 2
# x3 = np.cos(u3) * np.sin(v3)
# y3 = np.sin(u3) * np.sin(v3)
# z3 = np.cos(v3)
# ax.scatter(x2, y2, z2, color='b', s=0.5)


mean0 = [0.0, 0.0]
cov0 = [[0.3, 0], [0, 0.3]]
K = 50
u0, v0 = np.random.multivariate_normal(mean0, cov0, K).T
u0 = u0 * np.pi / 8 + np.pi * 3 / 4
v0 = v0 * np.pi / 8 + np.pi * 1 / 4
x0 = np.cos(u0) * np.sin(v0)
y0 = np.sin(u0) * np.sin(v0)
z0 = np.cos(v0)
# ax.scatter(x0, y0, z0, color='r', marker='o', s=2, facecolor=None)

plt.show()

x1 = np.stack((x1, y1, z1), axis=1).clip(-0.99, 0.99)
x2 = np.stack((x2, y2, z2), axis=1).clip(-0.99, 0.99)
# x3 = np.stack((x3, y3, z3), axis=1).clip(-0.99, 0.99)
x0 = np.stack((x0, y0, z0), axis=1).clip(-0.99, 0.99)

x0 = torch.from_numpy(x0)
x1 = torch.from_numpy(x1)
x2 = torch.from_numpy(x2)
# x3 = torch.from_numpy(x3)


use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


vwb = SVWB(x0, [x1, x2], device=device, verbose=False)
output = vwb.cluster(max_iter_h=5000, max_iter_p=1)
e_idx, pred_label_e = output['idx'], output['pred_label_e']

# scale p
vwb.data_p /= torch.norm(vwb.data_p, dim=1)[:, None]

xmin, xmax, ymin, ymax = -1.0, 1.0, -0.5, 0.5


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, color='gray', s=0.1)
# ax.plot_wireframe(x*0.95, y*0.95, z*0.95, color="lightgray")

for idx in [12]:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(x * 0.95, y * 0.95, z * 0.95, color="lightgray")
    colors = plt.cm.magma((x - x.min()) / float((x - x.min()).max()))
    ax.plot_surface(x * 0.95, y * 0.95, z * 0.95, antialiased=False, facecolors=colors, linewidth=0, shade=False)
    for i in range(2):
        ce = np.array(plt.get_cmap('viridis')(e_idx[i].cpu().numpy() / (K - 1)))

        ax.scatter(vwb.data_e[i][:, 0], vwb.data_e[i][:, 1], vwb.data_e[i][:, 2], s=1, color=ce, zorder=4)
    ax.scatter(vwb.data_p[:, 0], vwb.data_p[:, 1], vwb.data_p[:, 2], s=5, marker='o',
               facecolors='none', linewidth=2, color='r', zorder=5)

    e0s = vwb.data_e[0][e_idx[0] == idx]
    e1s = vwb.data_e[1][e_idx[1] == idx]
    p = vwb.data_p[idx]

    for e0, e1 in zip(e0s, e1s):
        x = [e1[0], p[0], e0[0]]
        y = [e1[1], p[1], e0[1]]
        z = [e1[2], p[2], e0[2]]
        plt.plot(x, y, z, c='gray', alpha=0.4, zorder=5)
    # ls = LightSource(azdeg=180, altdeg=45)
    ax.view_init(elev=10., azim=100.)
    plt.axis('off')
    # plt.savefig("4_5/sphere" + str(idx) + "test.svg", bbox_inches='tight')
    plt.savefig("sphere" + str(idx) + "test.png", dpi=300,  bbox_inches='tight')

