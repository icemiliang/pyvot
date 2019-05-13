from vot import VotAPNew
import utils
import time
import numpy as np
import matplotlib.pyplot as plt


ot = VotAPNew(ratio=500)
mean = [0, 0]
cov = [[.04, 0], [0, .04]]
data = np.random.multivariate_normal(mean, cov, 50).clip(-0.99, 0.99)
ot.import_data(data)
ot.map(sampling='unisquare', plot_filename='area_preserve.gif')

X_p_before = np.copy(ot.data_p_original)
fig = plt.figure(figsize=(5, 5))
plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True); plt.title('before')
plt.scatter(X_p_before[:, 0], X_p_before[:, 1], marker='o', color=utils.color_red, zorder=3)

img = utils.fig2data(fig)

img.save('test.png', 'png')
