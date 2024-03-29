# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Latest update: Sep 1st 2023
# Licence: MIT


import os
import sys
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_torch import VOT


np.random.seed(19)

x1 = np.loadtxt("kitten.csv", delimiter=',')

x1 = x1[:, [0, 2, 1]]


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


dim = 3            # number of dimensions of the points
noise_sigma = .0   # standard deviation error to be added
translation = 2    # max translation of the test set
rotation = .8      # max rotation (radians) of the test set

# Translate
t = np.random.rand(3) * translation
x2 = x1.copy() + t
R = rotation_matrix(np.random.rand(3), rotation)
x2 = np.dot(R, x2.T).T
num = x1.shape[0]
x2 += np.random.randn(num, 3) * noise_sigma

translation = -2   # max translation of the test set
rotation = -.6     # max rotation (radians) of the test set

t = np.random.rand(3) * translation
x3 = x1.copy() + t
R = rotation_matrix(np.random.rand(3), rotation)
x3 = np.dot(R, x3.T).T
x3 += np.random.randn(num, 3) * noise_sigma

translation = 0
rotation = .2

t = np.random.rand(3) * translation
x1 = x1.copy() + t
x1 += np.random.randn(num, 3) * noise_sigma

# scale
x1min = np.amin(x1, axis=0)
x1max = np.amax(x1, axis=0)
x2min = np.amin(x2, axis=0)
x2max = np.amax(x2, axis=0)
x3min = np.amin(x3, axis=0)
x3max = np.amax(x3, axis=0)

xmin = np.minimum(np.minimum(x1min[0], x2min[0]), x3min[0])
xmax = np.maximum(np.maximum(x1max[0], x2max[0]), x3max[0])
ymin = np.minimum(np.minimum(x1min[1], x2min[1]), x3min[1])
ymax = np.maximum(np.maximum(x1max[1], x2max[1]), x3max[1])
zmin = np.minimum(np.minimum(x1min[2], x2min[2]), x3min[2])
zmax = np.maximum(np.maximum(x1max[2], x2max[2]), x3max[2])

scale = np.maximum(xmax - xmin, np.maximum(ymax - ymin, zmax - zmin))

x1[:, 0] = 2 * (x1[:, 0] - xmin) / scale - 1
x1[:, 1] = 2 * (x1[:, 1] - ymin) / scale - 1
x1[:, 2] = 2 * (x1[:, 2] - zmin) / scale - 1
x2[:, 0] = 2 * (x2[:, 0] - xmin) / scale - 1
x2[:, 1] = 2 * (x2[:, 1] - ymin) / scale - 1
x2[:, 2] = 2 * (x2[:, 2] - zmin) / scale - 1
x3[:, 0] = 2 * (x3[:, 0] - xmin) / scale - 1
x3[:, 1] = 2 * (x3[:, 1] - ymin) / scale - 1
x3[:, 2] = 2 * (x3[:, 2] - zmin) / scale - 1

x1 = x1.clip(-0.999, 0.999)
x2 = x2.clip(-0.999, 0.999)
x3 = x3.clip(-0.999, 0.999)

dot_size = 0.5
dot_size_scale = 10
alpha = 0.3

color_map = np.array([[237, 125, 49, 255], [112, 173, 71, 255], [91, 155, 213, 255], [237, 41, 57, 255]]) / 255

downsample = 100
x1_down = x1[0::downsample, :]


# ------- run VOT -------- #

iterP = 8

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


x = torch.from_numpy(x1_down).to(device)
x1_copy = torch.from_numpy(x1).to(device)
x2_copy = torch.from_numpy(x2).to(device)
x3_copy = torch.from_numpy(x3).to(device)


vot = VOT(x, [x1_copy, x2_copy, x3_copy], device=device, verbose=False)
vot.cluster(lr=1, max_iter_h=3000, max_iter_y=iterP, lr_decay=500, beta=0.5, icp=True)

fig2 = plt.figure(figsize=(8, 8))
#
ax2 = fig2.add_subplot(111, projection='3d')
#
outE1 = vot.x[0].detach().cpu().numpy()
outE2 = vot.x[1].detach().cpu().numpy()
outE3 = vot.x[2].detach().cpu().numpy()
outP = vot.y.detach().cpu().numpy()

ax2.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=dot_size, color=color_map[0], alpha=alpha)
ax2.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=dot_size, color=color_map[1], alpha=alpha)
ax2.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=dot_size, color=color_map[2], alpha=alpha)

ax2.scatter(outE1[:, 0], outE1[:, 1], outE1[:, 2], s=dot_size, color=color_map[0], alpha=alpha)
ax2.scatter(outE2[:, 0], outE2[:, 1], outE2[:, 2], s=dot_size, color=color_map[1], alpha=alpha)
ax2.scatter(outE3[:, 0], outE3[:, 1], outE3[:, 2], s=dot_size, color=color_map[2], alpha=alpha)
ax2.scatter(outP[:, 0], outP[:, 1], outP[:, 2], s=5, marker='o',
               facecolors='none', linewidth=2, color=color_map[3], zorder=5)

bound = 1.
minx, maxx = -bound, bound
miny, maxy = -bound, bound
minz, maxz = -bound, bound

ax2.set_xlim(minx, maxx)
ax2.set_ylim(miny, maxy)
ax2.set_zlim(minz, maxz)

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.axis('off')

plt.savefig("kitten_{}_torch.png".format(iterP), bbox_inches='tight')
# plt.show()