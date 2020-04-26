import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vot_numpy import VOT
import utils

np.random.seed(19)

# ------------ Generate data ------------- #
mean1 = [0., -0.2]
cov1 = [[0.05, 0], [0, 0.05]]
x11, x12 = np.random.multivariate_normal(mean1, cov1, 1000).T
x1 = np.stack((x11, x12), axis=1).clip(-0.99, 0.99)

mean2 = [0.5, 0.5]
cov2 = [[0.01, 0], [0, 0.01]]
x21, x22 = np.random.multivariate_normal(mean2, cov2, 200).T
x2 = np.stack((x21, x22), axis=1).clip(-0.99, 0.99)

mean3 = [-0.5, 0.5]
cov3 = [[0.01, 0], [0, 0.01]]
x31, x32 = np.random.multivariate_normal(mean3, cov3, 200).T
x3 = np.stack((x31, x32), axis=1).clip(-0.99, 0.99)


mean = [0.0, 0.0]
cov = [[0.02, 0], [0, 0.02]]
K = 3
y1, y2 = np.random.multivariate_normal(mean, cov, K).T


xmin, xmax, ymin, ymax = -.7, .8, -.65, .8


y = np.stack((y1, y2), axis=1).clip(-0.99, 0.99)
x = np.concatenate((x1, x2, x3), axis=0)

plt.close()
# ---------------kmeans---------------

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init=y).fit(x)

label = kmeans.predict(x)
new_y = kmeans.cluster_centers_

color_map = np.array([[237, 125, 49, 255], [112, 173, 71, 255], [91, 155, 213, 255]]) / 255

device = 'cpu'

# ---------------VWB---------------
y = new_y

for reg in [0.5, 2, 1e9]:

    y_copy = y.copy()
    x_copy = x.copy()

    vot = VOT(y_copy, [x_copy], verbose=False)
    output = vot.cluster(lr=0.5, max_iter_h=1000, max_iter_y=1, beta=0.5, reg=reg)

    idx = output['idx']

    fig = plt.figure(figsize=(4, 4))
    ce = color_map[idx[0]]
    utils.scatter_otsamples(vot.y, vot.x[0], size_p=30, marker_p='o', color_e=ce,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    plt.axis('off')
    # plt.savefig("0.svg", bbox_inches='tight')
    plt.savefig(str(int(reg)) + ".png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(4, 4))

ce = color_map[label]
utils.scatter_otsamples(new_y, x, size_p=30, marker_p='o', color_e=ce,
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
plt.axis('off')
# plt.savefig("0.svg", bbox_inches='tight')
plt.savefig("0.png", dpi=300, bbox_inches='tight')
