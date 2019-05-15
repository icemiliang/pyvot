from vot import VotAP
import utils
import time
import numpy as np
import matplotlib.pyplot as plt


ot = VotAP(max_iter=200, ratio=1000)
mean = [0, 0]
cov = [[.04, 0], [0, .04]]
data = np.random.multivariate_normal(mean, cov, 20).clip(-0.99, 0.99)
ot.import_data(data)
# ot.map(sampling='unisquare')
# ot.map(sampling='unicircle', plot_filename='area_preserve.gif')
ot.map(sampling='unisquare', plot_filename='area_preserve.gif')

fig = plt.figure(figsize=(5, 5))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.title('Area-preserving mapping')
plt.scatter(ot.data_e[:, 0], ot.data_e[:, 1], s=0.2, color=utils.color_light_grey)
plt.scatter(ot.data_p_original[:, 0], ot.data_p_original[:, 1], s=2, marker='o', color=utils.color_blue)
plt.scatter(ot.data_p[:, 0], ot.data_p[:, 1], s=2, marker='o', color=utils.color_red)
plt.show()
