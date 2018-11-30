# Variational Optimal Transportation
# Author: Liang Mi <liangmi@asu.edu>
# Date: Nov 29th 2018

from vot import *
import matplotlib.pyplot as plt

# ------- set up -------- #
vot = Vot()
vot.import_data_file('data/p.csv','data/e.csv')
vot.setup(max_iter_p=1, max_iter_h=1500)

# ----- plot before ----- #
plt.subplot(121)
plt.scatter(vot.e_coor[:,0], vot.e_coor[:,1], marker='.')
plt.scatter(vot.p_coor[:,0], vot.p_coor[:,1], marker='o')

# ----- main program ---- #
vot.cluster()

# ----- plot after ------ #
plt.subplot(122)
plt.scatter(vot.e_coor[:,0], vot.e_coor[:,1], marker='.')
plt.scatter(vot.p_coor[:,0], vot.p_coor[:,1], marker='o')

# -------- plot --------- #
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
