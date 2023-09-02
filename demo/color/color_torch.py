# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Latest update: Sep 1st 2023
# Licence: MIT

import os
import sys
import torch
import numpy as np
import matplotlib
from mpl_toolkits import mplot3d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_torch import VOT
import imageio

np.random.seed(19)

# K = [16, 64, 256]
K = [16, 80, 336]
dot_size = 0.5
dot_size_scale = 50
alpha = 0.3

# -------------- load data ------------------ #

x1 = np.loadtxt("data/mountain1_array.csv", delimiter=',')
x2 = np.loadtxt("data/mountain2_array.csv", delimiter=',')
x3 = np.loadtxt("data/canyon_array.csv", delimiter=',')
x_all = np.loadtxt("data/all_array.csv", delimiter=',')

C1 = np.loadtxt("data/mountain1_C.csv", delimiter=',')
C2 = np.loadtxt("data/mountain2_C.csv", delimiter=',')
C3 = np.loadtxt("data/canyon_C.csv", delimiter=',')
C_all = np.loadtxt("data/all_C.csv", delimiter=',')

idx1 = np.loadtxt("data/mountain1_idx.csv", delimiter=',', dtype=int)
idx2 = np.loadtxt("data/mountain2_idx.csv", delimiter=',', dtype=int)
idx3 = np.loadtxt("data/canyon_idx.csv", delimiter=',', dtype=int)
idx_all = np.loadtxt("data/all_idx.csv", delimiter=',', dtype=int)


C1_16 = C1[:K[0], :]
C2_16 = C2[:K[0], :]
C3_16 = C3[:K[0], :]

idx1_16 = idx1[:, 0] - 1
idx2_16 = idx2[:, 0] - 1
idx3_16 = idx3[:, 0] - 1

C_all_16 = C_all[:K[0], :]
idx_all_16 = idx_all[:, 0] - 1


fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=dot_size, color=x1, alpha=alpha)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')
# fig1.savefig("x1_histogram.svg", bbox_inches='tight')
fig1.savefig("x1_histogram.png", dpi=300, bbox_inches='tight')

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=dot_size, color=x2, alpha=alpha)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
# fig2.savefig("x2_histogram.svg", bbox_inches='tight')
fig2.savefig("x2_histogram.png", dpi=300, bbox_inches='tight')

fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=dot_size, color=x3, alpha=alpha)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.set_xlabel('R')
ax3.set_ylabel('G')
ax3.set_zlabel('B')
# fig3.savefig("x3_histogram.svg", bbox_inches='tight')
fig3.savefig("x3_histogram.png", dpi=300, bbox_inches='tight')


# -------------- kmeans ------------------ #

x1_kmeans_16 = C1_16[idx1_16, :]
x2_kmeans_16 = C2_16[idx2_16, :]
x3_kmeans_16 = C3_16[idx3_16, :]

x1_kmeans_16 = np.transpose(np.reshape(x1_kmeans_16 * 255, (128, 128, 3)), (1, 0, 2))
x2_kmeans_16 = np.transpose(np.reshape(x2_kmeans_16 * 255, (128, 128, 3)), (1, 0, 2))
x3_kmeans_16 = np.transpose(np.reshape(x3_kmeans_16 * 255, (128, 128, 3)), (1, 0, 2))


imageio.imwrite("x1_kmeans_16.png", x1_kmeans_16.astype('uint8'))
imageio.imwrite("x2_kmeans_16.png", x2_kmeans_16.astype('uint8'))
imageio.imwrite("x3_kmeans_16.png", x3_kmeans_16.astype('uint8'))


fig4 = plt.figure(figsize=(8, 8))
ax4 = fig4.add_subplot(111, projection='3d')
ce1 = C1[idx1_16]
ax4.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=dot_size, color=ce1, alpha=alpha)
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False
ax4.scatter(C1_16[:, 0], C1_16[:, 1], C1_16[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax4.set_xlabel('R')
ax4.set_ylabel('G')
ax4.set_zlabel('B')
# fig4.savefig("x1_histogram_kmeans1.svg", bbox_inches='tight')
fig4.savefig("x1_histogram_kmeans1.png", dpi=300, bbox_inches='tight')

fig5 = plt.figure(figsize=(8, 8))
ax5 = fig5.add_subplot(111, projection='3d')
ce2 = C2[idx2_16]
ax5.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=dot_size, color=ce2, alpha=alpha)
ax5.xaxis.pane.fill = False
ax5.yaxis.pane.fill = False
ax5.zaxis.pane.fill = False
ax5.scatter(C2_16[:, 0], C2_16[:, 1], C2_16[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax5.set_xlabel('R')
ax5.set_ylabel('G')
ax5.set_zlabel('B')
# fig5.savefig("x2_histogram_kmeans1.svg", bbox_inches='tight')
fig5.savefig("x2_histogram_kmeans1.png", dpi=300, bbox_inches='tight')

fig6 = plt.figure(figsize=(8, 8))
ax6 = fig6.add_subplot(111, projection='3d')
ce3 = C3[idx3_16]
ax6.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=dot_size, color=ce3, alpha=alpha)
ax6.xaxis.pane.fill = False
ax6.yaxis.pane.fill = False
ax6.zaxis.pane.fill = False
ax6.scatter(C3_16[:, 0], C3_16[:, 1], C3_16[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax6.set_xlabel('R')
ax6.set_ylabel('G')
ax6.set_zlabel('B')
# fig6.savefig("x3_histogram_kmeans1.svg", bbox_inches='tight')
fig6.savefig("x3_histogram_kmeans1.png", dpi=300, bbox_inches='tight')

# -------------- VOT ------------------ #

use_gpu = False
if use_gpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

x1 = torch.from_numpy(x1.clip(-0.99, 0.99))
x2 = torch.from_numpy(x2.clip(-0.99, 0.99))
x3 = torch.from_numpy(x3.clip(-0.99, 0.99))


x = torch.from_numpy(C1_16.clip(-0.99, 0.99))
vot = VOT(x, [x1], device=device, verbose=False)
vot.cluster(lr=1, max_iter_h=5000, max_iter_y=1)
idx = vot.idx
x1_vot_16 = vot.y[idx[0], :].cpu().numpy()
x1_vot_16 = np.transpose(np.reshape(x1_vot_16 * 255, (128, 128, 3)), (1, 0, 2))
imageio.imwrite("x1_vot_16_torch.png", x1_vot_16.astype('uint8'))
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ce2 = vot.y[idx].numpy()
p11 = vot.y.numpy()
ax1.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=dot_size, color=ce2, alpha=alpha)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.scatter(p11[:, 0], p11[:, 1], p11[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')
# fig1.savefig("x1_histogram_vot_16_torch.svg", bbox_inches='tight')
fig1.savefig("x1_histogram_vot_16_torch.png", dpi=300, bbox_inches='tight')

x = torch.from_numpy(C2_16.clip(-0.99, 0.99))
vot = VOT(x, [x2], device=device, verbose=False)
vot.cluster(lr=1, max_iter_h=5000, max_iter_y=1)
idx = vot.idx
x2_vot_16 = vot.y[idx[0], :].cpu().numpy()
x2_vot_16 = np.transpose(np.reshape(x2_vot_16 * 255, (128, 128, 3)), (1, 0, 2))
imageio.imwrite("x2_vot_16.png", x2_vot_16.astype('uint8'))
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ce2 = vot.y[idx].numpy()
p21 = vot.y.numpy()
ax2.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=dot_size, color=ce2, alpha=alpha)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.scatter(p21[:, 0], p21[:, 1], p21[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
# fig2.savefig("x2_histogram_vot_16_torch.svg", bbox_inches='tight')
fig2.savefig("x2_histogram_vot_16_torch.png", dpi=300, bbox_inches='tight')

x = torch.from_numpy(C3_16.clip(-0.99, 0.99))
vot = VOT(x, [x3], device=device, verbose=False)
vot.cluster(lr=1, max_iter_h=5000, max_iter_y=1)
idx = vot.idx
x3_vot_16 = vot.y[idx[0], :].cpu().numpy()
x3_vot_16 = np.transpose(np.reshape(x3_vot_16 * 255, (128, 128, 3)), (1, 0, 2))
imageio.imwrite("x3_vot_16_torch.png", x3_vot_16.astype('uint8'))
fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(111, projection='3d')
ce3 = vot.y[idx].numpy()
p31 = vot.y.numpy()
ax3.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=dot_size, color=ce3, alpha=alpha)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.scatter(p31[:, 0], p31[:, 1], p31[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax3.set_xlabel('R')
ax3.set_ylabel('G')
ax3.set_zlabel('B')
# fig3.savefig("x3_histogram_vot_16.svg", bbox_inches='tight')
fig3.savefig("x3_histogram_vot_16_torch.png", dpi=300, bbox_inches='tight')


# -------------- VOT ALL ------------------ #

x = torch.from_numpy(C_all_16.clip(-0.99, 0.99))
vot = VOT(x, [x1, x2, x3], device=device, verbose=False)
vot.cluster(lr=1, max_iter_h=5000, max_iter_y=1)
idx = vot.idx


x1_vot_all = vot.y[idx[0], :].cpu().numpy()
x2_vot_all = vot.y[idx[1], :].cpu().numpy()
x3_vot_all = vot.y[idx[2], :].cpu().numpy()
x1_vot_all = np.transpose(np.reshape(x1_vot_all*255, (128, 128, 3)), (1, 0, 2))
x2_vot_all = np.transpose(np.reshape(x2_vot_all*255, (128, 128, 3)), (1, 0, 2))
x3_vot_all = np.transpose(np.reshape(x3_vot_all*255, (128, 128, 3)), (1, 0, 2))
imageio.imwrite("x1_vot_all_torch.png", x1_vot_all.astype('uint8'))
imageio.imwrite("x2_vot_all_torch.png", x2_vot_all.astype('uint8'))
imageio.imwrite("x3_vot_all_torch.png", x3_vot_all.astype('uint8'))

fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ce1 = vot.y[idx[0]].numpy()
p1 = vot.y.numpy()
ax1.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=dot_size, color=ce1, alpha=alpha)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')
# fig1.savefig("x1_histogram_vot1all_torch.svg", bbox_inches='tight')
fig1.savefig("x1_histogram_vot1all_torch.png", dpi=300, bbox_inches='tight')

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ce2 = vot.y[idx[1]].numpy()
p2 = vot.y.numpy()
ax2.scatter(x2[:, 0], x2[:, 1], x2[:, 2], s=dot_size, color=ce2, alpha=alpha)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
# fig2.savefig("x2_histogram_vot1all_torch.svg", bbox_inches='tight')
fig2.savefig("x2_histogram_vot1all_torch.png", dpi=300, bbox_inches='tight')

fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(111, projection='3d')
ce3 = vot.y[idx[2]].numpy()
p3 = vot.y.numpy()
ax3.scatter(x3[:, 0], x3[:, 1], x3[:, 2], s=dot_size, color=ce3, alpha=alpha)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.scatter(p3[:, 0], p3[:, 1], p3[:, 2], s=dot_size*dot_size_scale, color='k', zorder=1)
ax3.set_xlabel('R')
ax3.set_ylabel('G')
ax3.set_zlabel('B')
# fig3.savefig("x3_histogram_vot1all_torch.svg", bbox_inches='tight')
fig3.savefig("x3_histogram_vot1all_torch.png", dpi=300, bbox_inches='tight')
