# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Latest update: Sep 1st 2023
# Licence: MIT


import os
import sys
import numpy as np
import ot
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_numpy import VOT

np.random.seed(19)

# Generate data
mean1 = [-0.5, 0.]
cov1 = [[0.02, 0], [0, 0.02]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
x1 = np.stack((x1, y1), axis=1).clip(-0.99, 0.99)

mean2 = [0.5, 0.]
cov2 = [[0.02, 0], [0, 0.02]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
x2 = np.stack((x2, y2), axis=1).clip(-0.99, 0.99)

M = ot.dist(x1, x2, 'sqeuclidean')
G0 = ot.emd2([], [], M)
G1 = ot.sinkhorn2([], [], M, 1e-2)
print(G0)
print(G1.sum())

wd = np.zeros((50, 50))

k = 0
for K in [50, 125, 250, 500]:
    for i in range(5):
        mean = [0.0, -0.5]
        cov = [[0.02, 0], [0, 0.02]]
        x, y = np.random.multivariate_normal(mean, cov, K).T
        x = np.stack((x, y), axis=1).clip(-0.99, 0.99)

        vot = VOT(x, [x1, x2], verbose=False)
        output = vot.cluster(lr=1, max_iter_h=10000, max_iter_y=1, beta=0.9, lr_decay=500)
        wd[k, i] = output['wd']
        print(output['wd'])
        # np.savetxt(f, output['wd'], delimiter=',')
        del x
        del vot
    k += 1

# np.savetxt('ship_error/K_total.csv', wd, delimiter=',')
