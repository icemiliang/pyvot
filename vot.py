# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Dec 2nd 2018

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class Vot:
    """ variational optimal transportation """

    def setup(self, max_iter_h=1000, max_iter_p=10, thres=1e-8, rate=0.1):
        """ set up parameters

        Args:
            max_iter_h (int): max number of iterations of clustering
            max_iter_p (int): max number of iterations of transportation
            thres (float): threshold to break loops
            rate  (float): learning rate
        """

        self.h = np.zeros(self.np)
        self.thres = thres
        self.learnrate = rate
        self.max_iter_h = max_iter_h
        self.max_iter_p = max_iter_p

    def import_data_file(self, pfilename, efilename, mass=False, label=True):
        """ import data from files

        Args:
            pfilename (string): filename of p
            efilename (string): filename of e
            mass (bool): whether data has a mass column
        """
        p_data = np.loadtxt(open(pfilename, "r"), delimiter=",")
        e_data = np.loadtxt(open(efilename, "r"), delimiter=",")
        self.import_data(p_data, e_data, mass, label)

    def import_data(self, p_data, e_data, mass=False, label=True):
        """ import data from files

        Args:
            p_data np.ndarray(np,dim+): data of p, labels, mass, coordinates, etc
            e_data np.ndarray(ne,dim+): data of e, labels, mass, coordinates, etc
            mass (bool): whether data has a mass column
            label (bool): whether data has a label column
        """

        self.np = np.size(p_data, 0)
        self.ne = np.size(e_data, 0)

        if label and mass:
            self.p_label = p_data[:,0]
            self.e_label = e_data[:,0]
            self.p_dirac = p_data[:,1]
            self.e_mass = e_data[:,1]
            self.p_coor = p_data[:,2:]
            self.e_coor = e_data[:,2:]
        elif label and not mass:
            self.p_label = p_data[:,0]
            self.e_label = e_data[:,0]
            self.p_dirac = np.ones(self.np)/self.np
            self.e_mass = np.ones(self.ne)/self.ne
            self.p_coor = p_data[:,1:]
            self.e_coor = e_data[:,1:]
        elif not label and mass:
            self.p_label = -np.ones(self.ne)
            self.e_label = -np.ones(self.np)
            self.p_dirac = p_data[:, 1]
            self.e_mass = e_data[:, 1]
            self.p_coor = p_data[:, 1:]
            self.e_coor = e_data[:, 1:]
        else: # not label and not mass
            self.p_label = -np.ones(self.ne)
            self.e_label = -np.ones(self.np)
            self.p_dirac = np.ones(self.np)/self.np
            self.e_mass = np.ones(self.ne)/self.ne
            self.p_coor = p_data
            self.e_coor = e_data

        self.p_mass = np.zeros(self.np) # mass is the sum of its e's weights, its own weight is "dirac"
        self.p_label = self.p_label.astype(int)
        self.e_label = self.e_label.astype(int)

    def cluster(self, reg=0, alpha=0.01):
        """ compute Wasserstein clustering

        Args:
            reg int: flag for regularization, 0 means no regularization

        See Also
        --------
        update_p : update p
        update_map: computer optimal transportation
        """

        for iter_p in range(self.max_iter_p):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            for iter_h in range(self.max_iter_h):
                if self.update_map(iter_p,iter_h): break
            if self.update_p(iter_p, reg, alpha): break

    def update_map(self, iter_p, iter_h):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            float: max change of derivative, small max means convergence
        """

        # update dist matrix
        self.cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(self.cost, axis=0)
        # labels come from centroids
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.np):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        return True if np.amax(grad) < self.thres else False

    def update_p(self, iter_p, reg=0, alpha=0.01):
        """ update p

        Args:
            iter_p int: iteration index
            reg int or string: regularizer type
            alpha float: regularizer weight

        Returns:
            float: max change of p, small max means convergence
        """

        if reg == 1 or reg == 'potential':
            return self.update_p_reg_potential(iter_p, alpha)
        elif reg == 2 or reg == 'curvature':
            return self.update_p_reg_curvature(iter_p)
        else:
            return self.update_p_noreg(iter_p)

    def update_p_noreg(self, iter_p):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index

        Returns:
            float: max change of p, small max means convergence
        """

        max_change = 0.0
        # update p to the centroid of its clustered e samples
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.np):
            weight = self.e_mass[self.e_idx == j]
            if weight.size == 0:
                continue
            p_target = np.average(self.e_coor[self.e_idx == j,:], weights=weight, axis=0)
            # check if converge
            max_change = max(np.amax(self.p_coor[j,:] - p_target), max_change)
            self.p_coor[j,:] = p_target
        print("iter " + str(iter_p) + ": " + str(max_change))
        # return max p coor change
        return True if max_change < self.thres else False

    def update_p_reg_potential(self, iter_p, alpha=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            alpha float: regularizer weight

        Returns:
            float: max change of p, small max means convergence
        """

        def f(p, p0, label=None, alpha=0.01):
            """ objective function incorporating labels

            Args:
                p  np.array(np,dim):   p
                p0 np.array(np,dim):  centroids of e
                label np.array(np,): labels of p
                alpha float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + alpha*sum(delta*|pi-pj|^2), delta = 1 if li == lj
            """

            p = p.reshape(p0.shape)
            reg_term = 0.0
            # TODO Replace the nested for loop with matrix/vector operations, if possible
            for idx, l in np.ndenumerate(np.unique(label)):
                p_sub = p[label == l,:]
                for shift in range(1,np.size(p_sub,0)):
                    reg_term += np.sum(np.sum((p_sub-np.roll(p_sub, shift, axis=0))**2.0))

            return np.sum(np.sum((p - p0)**2.0)) + alpha*reg_term

        max_change = 0.0
        p0 = np.zeros((self.p_coor.shape))
        # new controid pos
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.np):
            weight = self.e_mass[self.e_idx == j]
            if weight.size == 0:
                continue
            p0[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=weight, axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - p0[j,:]),max_change)
        print("iter " + str(iter_p) + ": " + str(max_change))
        # regularize
        res = minimize(f, self.p_coor, method='BFGS', tol=self.thres, args=(p0,self.p_label,alpha))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(p0.shape)
        # return max change
        return True if max_change < self.thres else False
