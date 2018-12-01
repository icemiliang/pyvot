import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# def curvature(p1,p2,p3):
#     def compute_curvature(x1,y1,z1,x2,y2,z2,x3,y3,z3):
#         a = x1-2*x2+x3
#         b = y1-2*y2+y3
#         c = z1-2*z2+z3
#         dx = x1-x2
#         dy = y1-y2
#         dz = z1-z2
#
#         sq1 = c * dy - b * dz
#         sq2 = c * dx - a * dz
#         sq3 = b * dx - a * dy
#
#         numerator = sq1**2 + sq2**2 + sq3**2
#
#         def kp_ds (t):
#             sq4 = a * t - dx
#             sq5 = b * t - dy
#             sq6 = c * t - dz
#             return numerator/np.power(sq4**2 + sq5**2 + sq6**2, 2.5)
#
#         sum = 0
#         sum += kp_ds(0)/2.0
#         sum += kp_ds(1) / 2.0
#         for t in range(1,100):
#             t/=100.0
#             sum += kp_ds(t)
#         sum /= 100.0
#         return sum
#     return compute_curvature(p1[:,0],p1[:,1],p1[:,2],p2[:,0],p2[:,1],p2[:,2],p3[:,0],p3[:,1],p3[:,2])

def curvature(p2, x0, fix, alpha=0.1):
    # p = p.reshape(len(p)//9,3,3)
    # p1 = p[:,0,:]
    # p2 = p[:,1,:]
    # p3 = p[:,2,:]

    p1 = fix[0,:]
    p3 = fix[1,:]

    def compute_curvature(x1,y1,z1,x2,y2,z2,x3,y3,z3):
        a = x1-2*x2+x3
        b = y1-2*y2+y3
        c = z1-2*z2+z3
        dx = x1-x2
        dy = y1-y2
        dz = z1-z2
        numerator = (c*dy-b*dz)**2+(c*dx-a*dz)**2+(b*dx-a*dy)**2

        t = np.arange(0.0,1.01,0.01)
        k = numerator/np.power((a*t-dx)**2+(b*t-dy)**2+(c*t-dz)**2,2.5)
        k[0]/=2
        k[100] /= 2
        return np.sum(k)/100
    return np.sum(np.sum((x0[1,:]-p2)**2.0)) + \
           alpha * np.sum(compute_curvature(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p3[2]))
           # alpha*np.sum(compute_curvature(p1[:,0],p1[:,1],p1[:,2],p2[:,0],p2[:,1],p2[:,2],p3[:,0],p3[:,1],p3[:,2]))

# p = np.array([[0.0,0.0,0.0],
#               [1.0,1.0,0.0],
#               [2.0,0.0,0.0],
#               [3.0,2.0,0.0]])

# p = np.array([
#               [
#                 [0.0,0.0,0.0],
#                 [1.0,1.0,0.0],
#                 [2.0,0.0,0.0]
#               ],
#               [
#                 [1.0,1.0,0.0],
#                 [2.0,0.0,0.0],
#                 [3.0,1.0,0.0]
#               ]
#              ])

p = np.array([[0.0,0.0,0.0],
              [1.0,1.0,0.0],
              [2.0,0.0,0.0]])
x0 = p
fix = p[[0,2],:]
t0 = time.clock()
res = minimize(curvature, p[1,:], method='BFGS', tol=1e-8, args=(x0,fix))
print("time: " + str(time.clock()-t0) + " seconds\n")
x = res.x
# x = x.reshape(p.shape)
print(res)
print(x)
