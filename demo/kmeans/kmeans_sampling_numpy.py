# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: October 1st 2023
# Latest update: October 1st 2023
# Licence: MIT


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vot_numpy import VOT
import utils_numpy as utils

# np.random.seed(19)

# ------------ Generate data ------------- #
df = pd.read_csv('us_census.csv')
mask_48 = (df['state'] != 'AK') & (df['state'] != 'HI') & (df['state'] != 'PR')
df_48 = df[mask_48]
population_array = df_48['population'].to_numpy()
x = df_48[df.columns[2:4]].to_numpy()
x[:,0] = ((x[:,0] - np.min(x[:,0])) / (np.max(x[:,0]) - np.min(x[:,0])) - 0.5) * 1.99
x[:,1] = ((x[:,1] - np.min(x[:,1])) / (np.max(x[:,1]) - np.min(x[:,1])) - 0.5) * 1.99
K = 10

# ---------------kmeans--------------- #
index = np.random.choice(x.shape[0], K, replace=False)
init_centers = x[index, :]
kmeans = KMeans(n_clusters=K, init=init_centers).fit(x)
label = kmeans.predict(x)
y = kmeans.cluster_centers_

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].scatter(x[0::10,1], x[0::10,0], s=2, c='#009900', alpha=0.3)
axs[0].scatter(y[:,1], y[:,0], s=50, c='#0047AB', alpha=1, facecolors='none')
axs[0].axis([-1, 1, -1, 1])
# plt.show()
# ---------------VWB---------------
regs = [0.01, 0.1, 1e9]
for i in range(len(regs)):
    reg = regs[i]
    y_copy = init_centers.copy()
    x_copy = x.copy()

    vot = VOT(y_copy, [x_copy], verbose=False)
    vot.cluster(lr=0.5, max_iter_h=1000, max_iter_y=1, beta=0.5, reg=reg)

    # idx = vot.idx
    y = vot.y

    axs[i+1].scatter(x[0::10, 1], x[0::10, 0], s=2, c='#009900', alpha=0.3)
    axs[i+1].scatter(y[:, 1], y[:, 0], s=50, c='#0047AB', alpha=1, facecolors='none')
    axs[i+1].axis([-1, 1, -1, 1])

fig.savefig('test.jpeg', dpi='figure')
pass
