# Variational Optimal Transportation
# Author: Liang Mi <liangmi@asu.edu>
# Date: Nov 29th 2018

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

class Vot:
    def setup(self, max_iter_h=1000, max_iter_p=10, thres=1e-8, rate=0.1):
        self.h = np.zeros(self.p_num)
        self.thres = thres
        self.learnrate = rate
        self.max_iter_h = max_iter_h
        self.max_iter_p = max_iter_p
        self.cost = np.zeros((self.p_num,self.numE))
        self.e_idx = -np.ones(self.numE)
        self.e_predict = -np.ones(self.numE)

    def import_data_file(self, pfilename, efilename, mass=False):
        p_data = np.loadtxt(open(pfilename, "r"), delimiter=",")
        e_data = np.loadtxt(open(efilename, "r"), delimiter=",")
        self.import_data(p_data, e_data, mass)

    def import_data(self, p_data, e_data, mass=False):
        self.p_label = p_data[:,0]
        self.p_num = self.p_label.size
        self.p_coor = p_data[:,2:] if mass else p_data[:,1:]
        self.p_dirac = p_data[:,1] if mass else np.ones(self.p_num)/self.p_num
        self.p_mass = np.zeros(self.p_num)

        self.e_label = e_data[:,0]
        self.numE = self.e_label.size
        self.e_coor = e_data[:,2:] if mass else e_data[:,1:]
        self.e_mass = e_data[:,1] if mass else np.ones(self.numE)/self.numE

    def print_var(self):
        print("h: " + str(self.h))
        print("cost matrix: \n" + str(self.cost))
        print("p label: " + str(self.p_label))
        print("p dirac: " + str(self.p_dirac))
        print("p mass: " + str(self.p_mass))
        print("p coor: \n" + str(self.p_coor))
        print("e label: " + str(self.e_label))
        print("e predict: " + str(self.e_predict))
        print("e mass: " + str(self.e_mass))
        print("e idx: " + str(self.e_idx))
        print("e coor: \n" + str(self.e_coor))

    def cluster(self):
        for iter_p in range(self.max_iter_p):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            for iter_h in range(self.max_iter_h):
                if self.update_map(iter_p,iter_h): break
            if self.update_p(iter_p): break
        return False

    def update_map(self, iter_p, iter_h):
        # update dist matrix
        self.cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(self.cost, axis=0)
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.p_num):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        return True if np.amax(grad) < self.thres else False

    def update_p(self, iter_p):
        max_change = 0.0
        # update p to the centroid of its clustered e samples
        for j in range(self.p_num):
            tmp = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            # check if converge
            max_change = max(np.amax(self.p_coor[j,:] - tmp),max_change)
            self.p_coor[j,:] = tmp
        print("iter " + str(iter_p) + ": " + str(max_change))
        # return max p coor change
        return True if max_change < self.thres else False

    def f_potential(self, x, x0, label=None, alpha=0.5):
        x = x.reshape(x0.shape)
        return np.sum(np.sum((x0-x)**2.0)) + \
               alpha*np.sum(np.sum((x[1:,:]-x[:-1,:])**2.0) + (x[0,:]-x[-1,:])**2.0)

    # def f_curvature(self, p2, x0, fix, alpha1=0.1, alpha2=0.2):
    #     def compute_curvature(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    #         a = x1 - 2 * x2 + x3
    #         b = y1 - 2 * y2 + y3
    #         c = z1 - 2 * z2 + z3
    #         dx = x1 - x2
    #         dy = y1 - y2
    #         dz = z1 - z2
    #         numerator = (c*dy-b*dz)**2 + (c*dx-a*dz)**2 + (b*dx-a*dy)**2
    #
    #         def kp_ds(t):
    #             return numerator / np.power((a*t-dx)**2 + (b*t-dy)**2 + (c*t-dz)**2, 2.5)
    #
    #         sum = (kp_ds(0) + kp_ds(1)) / 2.0
    #         # Use matrix operations to replace for loop
    #         for t in range(1, 100):
    #             t /= 100.0
    #             sum += kp_ds(t)
    #         sum /= 100.0
    #         return sum
    #
    #     return np.sum(np.sum((x0[1, :] - p2) ** 2.0)) + \
    #            alpha2 * np.sum(compute_curvature(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]))

    def update_p_reg(self, iter_p):
        max_change = 0.0
        tmp = np.zeros((self.p_coor.shape))
        # new controid pos
        for j in range(self.p_num):
            tmp[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - tmp[j,:]),max_change)
        print("iter " + str(iter_p) + ": " + str(max_change))
        # regularize
        res = minimize(self.f_potential, self.p_coor, method='BFGS', tol=1e-8, args=(tmp,self.p_label))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(tmp.shape)
        # return max change
        return True if max_change < self.thres else False
