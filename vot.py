# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Nov 30th 2018

import numpy as np
from scipy.spatial.distance import cdist

class Vot:
    def setup(self, max_iter_h=1000, max_iter_p=10, thres=1e-8, rate=0.1):
        self.h = np.zeros(self.p_num)
        self.thres = thres
        self.learnrate = rate
        self.max_iter_h = max_iter_h
        self.max_iter_p = max_iter_p

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
        self.e_num = self.e_label.size
        self.e_coor = e_data[:,2:] if mass else e_data[:,1:]
        self.e_mass = e_data[:,1] if mass else np.ones(self.e_num)/self.e_num

    def cluster(self):
        for iter_p in range(self.max_iter_p):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            for iter_h in range(self.max_iter_h):
                if self.update_map(): break
            if self.update_p(iter_p): break
        return False

    def update_map(self):
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
