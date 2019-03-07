# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: MArch 6th 2019

import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from skimage import transform as tf

class VotAreaPreserve:
    """ Area Preserving with variational optimal transportation """

    def setup(self, max_iter = 2000, thres = 1e-8, rate = 0.2, ratio = 100, dim = 2):
        """ set up parameters

        Args:
            max_iter int: max number of iterations of optimal transportation
            thres float: threshold to break loops
            rate  float: learning rate
            ratio float: the ratio of num of e to the num of p
            dim     int: dimension of the data/space
        """

        self.thres = thres
        self.learnrate = rate
        self.max_iter = max_iter
        self.h = np.zeros(self.num_p)
        self.dim = dim
        self.ratio = ratio

        if self.dim < np.size(self.p_coor, 1):
            warnings.warn("Dimension of data larger than the setting.\n Truncating data...")
            self.p_coor = self.p_coor[:,0:self.dim]
        elif self.dim > np.size(self.p_coor, 1):
            warnings.warn("Dimension of data smaller than the setting.\n Resetting dim...")
            self.dim = np.size(self.p_coor, 1)

        assert np.amax(self.p_coor) < 1 and np.amin(self.p_coor) > -1, "Input data output boundary (-1, 1)."

    def import_data_from_file(self, pfilename, mass = False, label = True):
        """ import data from csv files

        Args:
            pfilename string: filename of p
            mass  bool: whether data has a mass column
            label bool: whether data has a label column

        See Also
        --------
        import_data : dump data into internal numpy arrays
        """

        p_data = np.loadtxt(open(pfilename, "r"), delimiter=",")

        if label and mass:
            self.import_data(p_data[:, 2:], yp = p_data[:, 0], mass_p = p_data[:, 1])
        elif label and not mass:
            self.import_data(p_data[:, 1:], yp = p_data[:, 0])
        elif not label and mass:
            self.import_data(p_data[:, 1:], mass_p = p_data[:, 0])
        else:
            self.import_data(p_data)

    def import_data(self, Xp, yp = None, mass_p = None):
        """ import data from numpy arrays

        Args:
            Xp np.ndarray(np,dim+): initial coordinates of p
            yp np.ndarray(num_p,): labels of p
            mass_p np.ndarray(num_p,): weights of p

        See Also
        --------
        import_data_file : import data from csv files
        """

        self.num_p = np.size(Xp, 0)
        self.p_label = yp.astype(int) if not yp is None else -np.ones(self.num_p).astype(int)
        self.p_dirac = mass_p if not mass_p is None else np.ones(self.num_p) / self.num_p
        self.p_coor = Xp

        # "p_mass" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.p_mass = np.zeros(self.num_p)

    def area_preserve(self):
        """ map p into the area

        :return:
        """
        self.random_sample()
        self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
        for iter in range(self.max_iter):
            if iter % 100 == 0:
                self.learnrate *= 0.95
            if self.update_map(iter): break
        self.update_p()

    def random_sample(self):
        """ randomly sample the area with dirac measures

        """
        pass

        self.num_e = self.num_p * self.ratio
        if self.num_e * self.dim > 1e8:
            warnings.warn("Sampling the area will take too much memory.")
        self.e_coor = np.random.random((self.num_e, self.dim)) * 2 - 1
        self.e_mass =  np.ones(self.num_e)/self.num_e
        self.e_label = -np.ones(self.num_e).astype(int)

    def update_map(self, iter):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        # update dist matrix
        cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(cost, axis = 0)
        # labels come from centroids
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.num_p):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        max_change = np.amax(grad)
        if iter % 200 == 0:
            print("iter %d: %.8f" % (iter, max_change))
        return True if max_change < self.thres else False

    def update_p(self):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index

        Returns:
            bool: convergence or not, determined by max p change
        """

        # update p to the centroid of its clustered e samples
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.num_p):
            idx_e_j = self.e_idx == j
            weights = self.e_mass[idx_e_j]
            if weights.size == 0:
                continue
            p_target = np.average(self.e_coor[idx_e_j,:], weights = weights, axis = 0)
            self.p_coor[j,:] = p_target

class Vot:
    """ variational optimal transportation """

    def setup(self, max_iter_h = 2000, max_iter_p = 10, thres = 1e-8, rate = 0.1):
        """ set up parameters

        Args:
            max_iter_h int: max number of iterations of clustering
            max_iter_p int: max number of iterations of transportation
            thres float: threshold to break loops
            rate  float: learning rate
        """

        self.thres = thres
        self.learnrate = rate
        self.max_iter_h = max_iter_h
        self.max_iter_p = max_iter_p
        self.h = np.zeros(self.num_p)

    def import_data_from_file(self, pfilename, efilename, mass = False, label = True):
        """ import data from csv files

        Args:
            pfilename string: filename of p
            efilename string: filename of e
            mass  bool: whether data has a mass column
            label bool: whether data has a label column

        See Also
        --------
        import_data : dump data into internal numpy arrays
        """

        p_data = np.loadtxt(open(pfilename, "r"), delimiter = ",")
        e_data = np.loadtxt(open(efilename, "r"), delimiter = ",")

        if label and mass:
            self.import_data(p_data[:,2:], e_data[:,2:],
                             yp = p_data[:,0], ye = e_data[:,0],
                             mass_p = p_data[:,1], mass_e = e_data[:,0])
        elif label and not mass:
            self.import_data(p_data[:, 1:], e_data[:, 1:],
                             yp = p_data[:, 0], ye = e_data[:, 0])
        elif not label and mass:
            self.import_data(p_data[:, 1:], e_data[:, 1:],
                             mass_p = p_data[:, 0], mass_e = e_data[:, 0])
        else:
            self.import_data(p_data, e_data)

    def import_data(self, Xp, Xe, yp = None, ye = None, mass_p = None, mass_e = None):
        """ import data from numpy arrays

        Args:
            Xp np.ndarray(np,dim+): initial coordinates of p
            Xe np.ndarray(ne,dim+): coordinates of e
            yp np.ndarray(num_p,): labels of p
            ye np.ndarray(num_e,): initial labels of e
            mass_p np.ndarray(num_p,): weights of p
            mass_e np.ndarray(num_e,): weights of e

        See Also
        --------
        import_data_file : import data from csv files
        """

        self.num_p = np.size(Xp, 0)
        self.num_e = np.size(Xe, 0)
        
        self.p_label = yp.astype(int) if not yp is None else -np.ones(self.num_p).astype(int)
        self.e_label = ye.astype(int) if not yp is None else -np.ones(self.num_e).astype(int)

        self.p_dirac = mass_p if not mass_p is None else np.ones(self.num_p)/self.num_p
        self.e_mass = mass_e if not mass_e is None else np.ones(self.num_e)/self.num_e

        self.p_coor = Xp
        self.e_coor = Xe

        # "p_mass" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.p_mass = np.zeros(self.num_p)

        if abs(np.sum(self.p_dirac) - np.sum(self.e_mass)) > 1e-6:
            warnings.warn("Total mass of e does not equal to total mass of p")

    def cluster(self, reg_type = 0, reg = 0.01):
        """ compute Wasserstein clustering

        Args:
            reg int: flag for regularization, 0 means no regularization

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        for iter_p in range(self.max_iter_p):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            for iter_h in range(self.max_iter_h):
                if self.update_map(iter_p,iter_h): break
            if self.update_p(iter_p, reg_type, reg): break

    def update_map(self, iter_p, iter_h):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        # update dist matrix
        cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(cost, axis = 0)
        # labels come from centroids
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.num_p):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        return True if np.amax(grad) < self.thres else False

    def update_p(self, iter_p, reg_type = 0, reg = 0.01):
        """ update p

        Args:
            iter_p int: iteration index
            reg int or string: regularizer type
            reg float: regularizer weight

        Returns:
            float: max change of p, small max means convergence
        """

        if reg_type == 1 or reg_type == 'potential':
            return self.update_p_reg_potential(iter_p, reg)
        elif reg_type == 2 or reg_type == 'transform':
            return self.update_p_reg_transform(iter_p, reg)
        else:
            return self.update_p_noreg(iter_p)

    def update_p_noreg(self, iter_p):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index

        Returns:
            bool: convergence or not, determined by max p change
        """

        max_change = 0.0
        # update p to the centroid of its clustered e samples
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.num_p):
            idx_e_j = self.e_idx == j
            weights = self.e_mass[idx_e_j]
            if weights.size == 0:
                continue
            p_target = np.average(self.e_coor[idx_e_j,:], weights = weights, axis = 0)
            # check if converge
            max_change = max(np.amax(self.p_coor[j,:] - p_target), max_change)
            self.p_coor[j,:] = p_target
        print("iter %d: %.8f" % (iter_p, max_change))
        # return max p coor change
        return True if max_change < self.thres else False

    def update_p_reg_potential(self, iter_p, reg = 0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, label = None, reg = 0.01):
            """ objective function incorporating labels

            Args:
                p  np.array(np,dim):   p
                p0 np.array(np,dim):  centroids of e
                label np.array(np,): labels of p
                reg float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + reg * sum(1(li == lj)*|pi-pj|^2)
            """

            p = p.reshape(p0.shape)
            reg_term = 0.0
            for idx, l in np.ndenumerate(np.unique(label)):
                p_sub = p[label == l,:]
                # pairwise distance with smaller memory burden
                # |pi - pj|^2 = pi^2 + pj^2 - 2*pi*pj
                reg_term += np.sum((p_sub ** 2).sum(axis = 1, keepdims = True) + \
                                   (p_sub ** 2).sum(axis = 1) - \
                                   2 * p_sub.dot(p_sub.T))

            return np.sum((p - p0)**2.0) + reg * reg_term

        if (np.unique(self.p_label).size == 1): warnings.warn("All known samples belong to the same class")

        max_change = 0.0
        p0 = np.zeros((self.p_coor.shape))

        # new controid pos
        for j in range(self.num_p):
            idx_e_j = self.e_idx == j
            weight = self.e_mass[idx_e_j]
            if weight.size == 0:
                continue
            p0[j,:] = np.average(self.e_coor[idx_e_j,:], weights = weight, axis = 0)
            max_change = max(np.amax(self.p_coor[j,:] - p0[j,:]),max_change)
        print("iter %d: %.8f" % (iter_p, max_change))

        # regularize
        res = minimize(f, self.p_coor, method = 'BFGS', tol = self.thres, args = (p0, self.p_label, reg))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(p0.shape)
        # return max change
        return True if max_change < self.thres else False

    def update_p_reg_transform(self, iter_p, reg = 0.01):
        """ update each p to the centroids of its cluster,
            regularized by an affine transformation 
            which is estimated from the OT map.

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, pa, reg = 0.01):
            """ objective function regularized by affine transformations

            Args:
                p  np.array(np,dim): p
                p0 np.array(np,dim): centroids of e
                pa np.array(np,dim): target position of p after affine
                reg float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + reg * sum(|p-pa|^2)
            """
            p = p.reshape(p0.shape)
            return np.sum((p-p0)**2.0) + reg * np.sum((p-pa)**2.0)

        # assert self.dim == 2, "dim has to equal 2"

        max_change = 0.0
        p0 = np.zeros((self.p_coor.shape))
        # new controid pos
        for j in range(self.num_p):
            idx_e_j = self.e_idx == j
            weight = self.e_mass[idx_e_j]
            if weight.size == 0:
                continue
            p0[j,:] = np.average(self.e_coor[idx_e_j,:], weights = weight, axis = 0)
            max_change = max(np.amax(self.p_coor[j,:] - p0[j,:]),max_change)
        print("iter %d: %.8f" % (iter_p, max_change))

        pa = np.zeros(p0.shape)

        for idx, l in np.ndenumerate(np.unique(self.p_label)):
            idx_p_label = self.p_label == l
            p_sub = self.p_coor[idx_p_label,:]
            p0_sub = p0[idx_p_label,:]
            # TODO estimating a high-dimensional transformation is a todo
            T = tf.EuclideanTransform()
            # T = tf.AffineTransform()
            # T = tf.ProjectiveTransform()
            T.estimate(p_sub, p0_sub)
            pa[idx_p_label,:] = T(p_sub)

        res = minimize(f, self.p_coor, method = 'BFGS', tol = self.thres, args = (p0, pa, reg))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(p0.shape)
        # return max change
        return True if max_change < self.thres else False
