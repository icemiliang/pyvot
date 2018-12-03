import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

skel = np.loadtxt(open('data/vent/skel.csv', "r"), delimiter=",")

gamma1 = [np.linspace(skel[0,i],skel[3,i],num=10) for i in range(3)]
gamma1 = np.array(gamma1)

gamma2 = [np.linspace(skel[1,i],skel[3,i],num=10) for i in range(3)]
gamma2 = np.array(gamma2)

gamma3 = [np.linspace(skel[2,i],skel[3,i],num=10) for i in range(3)]
gamma3 = np.array(gamma3)

np.savetxt("data/vent/gamma1.csv", gamma1.transpose(), fmt = '%.1f', delimiter=",")
np.savetxt("data/vent/gamma2.csv", gamma2.transpose(), fmt = '%.1f', delimiter=",")
np.savetxt("data/vent/gamma3.csv", gamma3.transpose(), fmt = '%.1f', delimiter=",")

vent = np.loadtxt(open('data/vent/m5.csv', "r"), delimiter=",")

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.axis('equal')

ax.plot3D(gamma1[0,:], gamma1[1,:], gamma1[2,:], 'gray')
ax.scatter3D(gamma1[0,:], gamma1[1,:], gamma1[2,:], c=gamma1[2,:], cmap='Greens')

ax.plot3D(gamma2[0,:], gamma2[1,:], gamma2[2,:], 'gray')
ax.scatter3D(gamma2[0,:], gamma2[1,:], gamma2[2,:], c=gamma2[2,:], cmap='Greens')

ax.plot3D(gamma3[0,:], gamma3[1,:], gamma3[2,:], 'gray')
ax.scatter3D(gamma3[0,:], gamma3[1,:], gamma3[2,:], c=gamma3[2,:], cmap='Greens')

ax.scatter3D(vent[:,0], vent[:,1], vent[:,2], c=vent[:,2], cmap='Greens',s=0.1)

plt.show()
