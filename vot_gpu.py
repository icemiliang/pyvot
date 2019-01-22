# PyVot
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Jan 18th 2019

import warnings

import numpy as np
import cupy


class VotAreaPreserveGPU:
    """ Area Preserving with variational optimal transportation """

    def setup(self, max_iter = 3000, thres = 1e-8, rate = 0.2, ratio = 100, dim = 2, verbose = True):
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
        self.h = cupy.zeros(self.num_p)
        self.dim = dim
        self.ratio = ratio
        self.verbose = verbose

        if self.dim < cupy.size(self.p_coor, 1):
            warnings.warn("Dimension of data larger than the setting.\n Truncating data...")
            self.p_coor = self.p_coor[:,0:self.dim]
        elif self.dim > cupy.size(self.p_coor, 1):
            warnings.warn("Dimension of data smaller than the setting.\n Resetting dim...")
            self.dim = cupy.size(self.p_coor, 1)

        # assert np.amax(self.p_coor) < 1 and np.amin(self.p_coor) > -1, "Input data output boundary (-1, 1)."

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

        tmp = np.loadtxt(open(pfilename, "r"), delimiter=",")
        p_data = cupy.array(tmp)

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

        self.num_p = cupy.size(Xp, 0)
        self.p_label = yp.astype(int) if not yp is None else -cupy.ones(self.num_p).astype(int)
        self.p_dirac = mass_p if not mass_p is None else cupy.ones(self.num_p) / self.num_p
        self.p_coor = Xp

        # "p_mass" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.p_mass = cupy.zeros(self.num_p)

    def area_preserve(self):
        """ map p into the area

        :return:
        """
        self.random_sample()
        # self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
        self.cost_base = cupy.transpose(cupy.sum((self.p_coor[None, :] - self.e_coor[:, None]) ** 2, -1))
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
        self.e_coor = cupy.random.random((self.num_e, self.dim)) * 2 - 1
        self.e_mass =  cupy.ones(self.num_e)/self.num_e
        self.e_label = -cupy.ones(self.num_e).astype(int)

    def update_map(self, iter):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        # update dist matrix
        cost = self.cost_base - self.h[:, cupy.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = cupy.argmin(cost, axis = 0)
        # labels come from centroids
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.num_p):
            self.p_mass[j] = cupy.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        max_change = cupy.amax(grad)
        if self.verbose and iter % 200 == 0:
            print("iter " + str(iter) + ": " + str(max_change))
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
            p_target = cupy.average(self.e_coor[idx_e_j,:], weights = weights, axis = 0)
            self.p_coor[j,:] = p_target
