# Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Nov 30th 2018

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

    def import_data_file(self, pfilename, efilename, mass=False):
        """ import data from files

        Args:
            pfilename (string): filename of p
            efilename (string): filename of e
            mass (bool): whether data has a mass column
        """

        p_data = np.loadtxt(open(pfilename, "r"), delimiter=",")
        e_data = np.loadtxt(open(efilename, "r"), delimiter=",")
        self.import_data(p_data, e_data, mass)

    def import_data(self, p_data, e_data, mass=False):
        """ import data from files

        Args:
            p_data np.ndarray(np,dim+): data of p, labels, mass, coordinates, etc
            e_data np.ndarray(ne,dim+): data of e, labels, mass, coordinates, etc
            mass (bool): whether data has a mass colum
        """

        self.p_label = p_data[:,0]
        self.np = self.p_label.size
        self.p_coor = p_data[:,2:] if mass else p_data[:,1:]
        self.p_dirac = p_data[:,1] if mass else np.ones(self.np)/self.np
        self.p_mass = np.zeros(self.np)

        self.e_label = e_data[:,0]
        self.ne = self.e_label.size
        self.e_coor = e_data[:,2:] if mass else e_data[:,1:]
        self.e_mass = e_data[:,1] if mass else np.ones(self.ne)/self.ne

    def print_var(self):
        """ print attributes to console """

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

    def cluster(self, reg=0):
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
            if self.update_p(iter_p, reg): break

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
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.np):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        return True if np.amax(grad) < self.thres else False

    def update_p(self, iter_p, reg=0):
        if reg == 1:
            return self.update_p_reg_potential(iter_p)
        elif reg == 2:
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
        for j in range(self.np):
            tmp = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            # check if converge
            max_change = np.amax(np.amax(self.p_coor[j,:] - tmp), max_change)
            self.p_coor[j,:] = tmp
        print("iter " + str(iter_p) + ": " + str(max_change))
        # return max p coor change
        return True if max_change < self.thres else False

    def update_p_reg_potential(self, iter_p):
        def f(p, p0, label=None, alpha=0.1):
            """ objective function incorporating labels

            Args:
                p  np.array(np,dim):   p
                p0 np.array(np,dim):  centroids of e
                label np.array(np,): labels of p
                alpha float: regularizer weight

            Returns:
                float: f = sum(|x-x0|^2) + alpha*sum(|xi-xj|^2)
            """

            p = p.reshape(p0.shape)
            return np.sum(np.sum((p0 - p)**2.0)) + \
                   alpha * np.sum(np.sum((p[1:,:] - p[:-1,:])**2.0) + (p[0,:] - p[-1,:])**2.0)

        max_change = 0.0
        tmp = np.zeros((self.p_coor.shape))
        # new controid pos
        for j in range(self.np):
            tmp[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - tmp[j,:]),max_change)
        print("iter " + str(iter_p) + ": " + str(max_change))
        # regularize
        res = minimize(f, self.p_coor, method='BFGS', tol=1e-8, args=(tmp,self.p_label))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(tmp.shape)
        # return max change
        return True if max_change < self.thres else False

    def update_p_reg_curvature(self, iter_p):
        def f(p, p0, pfix, alpha1=0.1, alpha2=0.1):
            """ objective function incorporating length and curvature

            Args:
                p  np.ndarray(np-pfix_num,dim): p
                p0 np.ndarray(np-pfix_num,dim): centroids of e
                pfix np.ndarray(fix_num,): labels of p
                alpha1 float: weight of regularizer length
                alpha2 float: weight of regularizer curvature

            Returns:
                float: f = sum(|x-x0|^2) + alpha1*sum(length(xj,xj-1) + alpha2*sum(curvature(xj,xj-1,xj-2)))
            """

            def length(x1,y1,z1,x2,y2,z2,x3,y3,z3):
                a = x1 - 2 * x2 + x3
                b = y1 - 2 * y2 + y3
                c = z1 - 2 * z2 + z3
                dx = x1 - x2
                dy = y1 - y2
                dz = z1 - z2

                t = np.array(0.0,1.01,0.01)
                ds = np.sqrt((a*t - dx)**2 + (b*t - dy)**2 + (c*t - dz)**2)
                return np.sum(ds)

            def curvature(x1, y1, z1, x2, y2, z2, x3, y3, z3):
                a = x1 - 2 * x2 + x3
                b = y1 - 2 * y2 + y3
                c = z1 - 2 * z2 + z3
                dx = x1 - x2
                dy = y1 - y2
                dz = z1 - z2
                numerator = (c*dy - b*dz)**2 + (c*dx - a*dz)**2 + (b*dx - a*dy)**2

                t = np.arange(0.0, 1.01, 0.01)
                k = numerator / np.power((a*t - dx)**2 + (b*t - dy)**2 + (c*t - dz)**2, 2.5)
                k[0] /= 2
                k[100] /= 2
                return np.sum(k)/100

            cost = 0
            cost +=  np.sum(np.sum((p0 - p)**2.0)) + \
                     alpha1 * np.sum(length())+\
                     alpha2 * np.sum(curvature())
            return cost
