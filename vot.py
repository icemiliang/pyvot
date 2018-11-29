# Variational Optimal Transportation
#
# Author: Liang Mi <liangmi@asu.edu>
#
# Date: Nov 28th 2018
#
# License: MIT License

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

class Vot:
    def setup(self, maxIterH=1000, maxIterP=10, thres=1e-8, rate=0.1):
        self.h = np.zeros(self.numP)
        self.thres = thres
        self.learnrate = rate
        self.maxIterH = maxIterH
        self.maxIterP = maxIterP
        self.cost = np.zeros((self.numP,self.numE))
        self.e_idx = -np.ones(self.numE)
        self.e_predict = -np.ones(self.numE)

    def import_data_file(self, pfilename, efilename, mass=False):
        dataP = np.loadtxt(open(pfilename, "r"), delimiter=",")
        dataE = np.loadtxt(open(efilename, "r"), delimiter=",")
        self.import_data(dataP, dataE, mass)

    def import_data(self, dataP, dataE, mass=False):
        self.p_label = dataP[:,0]
        self.numP = self.p_label.size
        self.p_coor = dataP[:,1:] if not mass else  dataP[:,2:]
        self.p_dirac = np.ones(self.numP)/self.numP if not mass else dataP[:,1]
        self.p_mass = np.zeros(self.numP)

        self.e_label = dataE[:, 0]
        self.numE = self.e_label.size
        self.e_coor = dataE[:, 1:] if not mass else dataE[:, 2:]
        self.e_mass = np.ones(self.numE)/self.numE if not mass else dataE[:,1]

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
        for iterP in range(self.maxIterP):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            # t0 = time.clock()
            for iterH in range(self.maxIterH):
                if self.update_map(iterP,iterH): break
            # print("map time: " + str(time.clock() - t0) + " seconds")
            if self.update_p_reg(iterP): break
        return False

    def update_p(self, iterP):
        max_change = 0.0
        for j in range(self.numP):
            tmp = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - tmp),max_change)
            self.p_coor[j, :] = tmp
        print("iter " + str(iterP) + ": " + str(max_change))
        return True if max_change < self.thres else False

    def update_map(self, iterP, iterH):
        # update dist matrix
        self.cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(self.cost, axis=0)
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.numP):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # return max
        return True if np.amax(grad) < self.thres else False

    def f_reg_potential(self, x0, x1, alpha=0.5):
        x0 = x0.reshape(x1.shape)
        return np.sum(np.sum((x1-x0)**2.0)) + \
               alpha*np.sum(np.sum((x0[1:,:]-x0[:-1,:])**2.0) + (x0[0,:]-x0[-1,:])**2.0)

    def update_p_reg(self, iterP):
        # t0 = time.clock()
        max_change = 0.0
        tmp = np.zeros((self.p_coor.shape))
        for j in range(self.numP):
            tmp[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - tmp[j,:]),max_change)
        print("iter " + str(iterP) + ": " + str(max_change))
        res = minimize(self.f_reg_potential, self.p_coor, method='BFGS', tol=1e-8, args=tmp)
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(tmp.shape)
        # print("p time: " + str(time.clock() - t0) + " seconds")
        print(res)
        return True if max_change < self.thres else False
