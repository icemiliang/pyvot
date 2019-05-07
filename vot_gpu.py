# PyVot
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Jan 18th 2019

import warnings

import numpy as np
import torch
import torch.nn.functional as F

class VotAreaPreserve:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the empirical samples

    def __init__(self, max_iter=2000, thres=1e-5, lr=0.2, ratio=100, dim=2, verbose=True):
        """ set up parameters
        Args:
            max_iter int: max number of iterations of optimal transportation
            thres float: threshold to break loops
            rate  float: learning rate
            ratio float: the ratio of num of e to the num of p
            dim     int: dimension of the data/space

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            ratio    float: ratio of num_e to num_p
            h        float: VOT optimizer, "height vector
            verbose   bool: console output verbose flag
            max_iter   int: maximum iteration
            num_p      int: number of p
            num_e      int: number of e
            dim        int: dimension of X
            X_p    numpy ndarray: coordinates of p
            y_p    numpy ndarray: labels of p
            mass_p numpy ndarray: mass of clusters of p

        """
        self.thres = thres
        self.lr = lr
        self.max_iter = max_iter
        self.h = None
        self.verbose = verbose
        self.X_p = None
        self.y_p = None
        self.num_p = None
        self.num_e = None
        self.p_dirac = None
        self.mass_p = None
        self.X_e = None
        self.dim = dim
        self.ratio = ratio

    def import_data_from_file(self, filename, has_mass=False, has_label=False):
        """ import data from a csv file

        Args:
            filename string: file name of p
            has_mass  bool: whether data has a has_mass column
            has_label bool: whether data has a label column

        See Also
        --------
        import_data : dump data into internal numpy arrays
        """

        data = np.loadtxt(filename, delimiter=",")
        data = torch.from_numpy(data)

        if has_label and has_mass:
            self.import_data(data[:, 2:], y_p=data[:, 0], mass_p=data[:, 1])
        elif has_label and not has_mass:
            self.import_data(data[:, 1:], y_p=data[:, 0])
        elif not has_label and has_mass:
            self.import_data(data[:, 1:], mass_p=data[:, 0])
        else:
            self.import_data(data)

    def import_data(self, X_p, y_p=None, mass_p=None):
        """ import data from numpy arrays

        Args:
            Xp np.ndarray(np,dim+): initial coordinates of p
            yp np.ndarray(num_p,): labels of p
            mass_p np.ndarray(num_p,): weights of p

        See Also
        --------
        import_data_file : import data from csv files
        """

        self.num_p = X_p.shape[0]
        self.y_p = y_p.type(torch.long) if not y_p is None else -torch.ones(self.num_p).type(torch.long)
        self.p_dirac = mass_p if not mass_p is None else torch.ones(self.num_p).type(torch.float) / self.num_p
        self.X_p = X_p

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.mass_p = torch.zeros(self.num_p)

        assert self.dim == self.X_p.shape[1], "Dimension of data not equal to the setting"

        assert self.X_p.max() < 1 and self.X_p.min() > -1, "Input output boundary (-1, 1)."
        self.h = torch.zeros(self.num_p)

    def area_preserve(self):
        """ map p into the area

        :return:
        """
        self.random_sample()
        self.cost_base = F.pairwise_distance(self._p, self.X_e, 2) ** 2
        for iter in range(self.max_iter):
            if iter != 0 and iter % 100 == 0:
                self.lr *= 0.95
            if self.update_map(iter): break
        self.update_p()

    def random_sample(self):
        """ randomly sample the area with dirac measures

        """
        self.num_e = self.num_p * self.ratio
        if self.num_e * self.dim > 1e8:
            warnings.warn("Sampling the area will take too much memory.")
        self.X_e = torch.rand(self.num_e, self.dim) * 2 - 1
        self.mass_e = torch.ones(self.num_e) / self.num_e
        self.y_e = -torch.ones(self.num_e).type(torch.long)

    def update_map(self, iter):
        """ update each p to the centroids of its cluster

        Args:
            iter int: iteration index of optimal transport

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        # update dist matrix
        cost = self.cost_base - self.h[:, None]
        # find nearest p for each e and add mass to p
        self.e_idx = torch.argmin(cost, dim=0)
        # labels come from centroids
        self.e_predict = self.y_p[self.e_idx]
        for j in range(self.num_p):
            self.mass_p[j] = torch.sum(self.mass_e[self.e_idx == j])
        # update gradient and h
        grad = self.mass_p - self.p_dirac
        self.h = self.h - self.lr * grad
        # check if converge and return max derivative
        max_change = torch.max(grad)
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
            weights = self.mass_e[idx_e_j]
            if weights.size == 0:
                continue
            p_target = torch.mean(self.X_e[idx_e_j, :] * weights, axis = 0)
            self.X_p[j, :] = p_target
