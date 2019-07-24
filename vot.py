# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 30th 2019


import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from skimage import transform as tf
import imageio
import utils


class Vot:
    """ variational optimal transportation """

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 mass_p=None, mass_e=None, thres=1e-5, verbose=True):
        """ set up parameters

        Args:
            thres float: threshold to break loops
            data_p numpy ndarray: initial coordinates of p
            label_p numpy ndarray: labels of p
            mass_p numpy ndarray: weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            verbose   bool: console output verbose flag
            num_p      int: number of p
            data_p     numpy ndarray: coordinates of p
            data_e     numpy ndarray: coordinates of e
            label_p    numpy ndarray: labels of p
            label_e    numpy ndarray: labels of e
            mass_p     numpy ndarray: mass of clusters of p
            mass_e     numpy ndarray: mass of e
            dirac_p    numpy ndarray: dirac measure of p
        """

        if not isinstance(data_p, np.ndarray):
            raise Exception('data_p is not a numpy ndarray')
        if not isinstance(data_e, np.ndarray):
            raise Exception('data_e is not a numpy ndarray')
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.copy()
        self.data_e_original = self.data_e.copy()

        num_p = data_p.shape[0]
        num_e = data_e.shape[0]

        if label_p is not None and not isinstance(label_p, np.ndarray):
            raise Exception('label_p is not a numpy ndarray')
        if label_e is not None and not isinstance(label_e, np.ndarray):
            raise Exception('label_e is not a numpy ndarray')
        self.label_p = label_p
        self.label_e = label_e

        if mass_p is not None and not isinstance(mass_p, np.ndarray):
            raise Exception('label_p is not a numpy ndarray')
        if mass_p is not None:
            self.dirac_p = mass_p
        else:
            self.dirac_p = np.ones(num_p) / num_p

        self.thres = thres
        self.verbose = verbose

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "dirac_p"
        self.mass_p = np.zeros(num_p)

        self.dirac_p = mass_p if mass_p is not None else np.ones(num_p) / num_p
        self.mass_e = mass_e if mass_e is not None else np.ones(num_e) / num_e

        assert np.max(self.data_p) <= 1 and np.min(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

    def cluster(self, reg_type=0, reg=0.01, lr=0.2, max_iter_p=10, max_iter_h=2000, lr_decay=200):
        """ compute Wasserstein clustering

        Args:
            reg_type   int: specify regulazation term, 0 means no regularization
            reg        int: regularization weight
            max_iter_p int: max num of iteration of clustering
            max_iter_h int: max num of updating h
            lr       float: GD learning rate
            lr_decay float: learning rate decay

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        for iter_p in range(max_iter_p):
            dist = cdist(self.data_p, self.data_e) ** 2
            self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay)
            if self.update_p(iter_p, reg_type, reg):
                break

    def update_map(self, dist, max_iter=3000, lr=0.2, beta=0.9, lr_decay=200):
        """ update each p to the centroids of its cluster

        Args:
            dist    numpy ndarray: dist matrix across p and e
            max_iter   int: max num of iterations
            lr       float: gradient descent learning rate
            beta     float: GD momentum
            lr_decay float: learning rate decay

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        num_p = self.data_p.shape[0]
        dh = 0

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            self.e_idx = np.argmin(dist, axis=0)
            self.mass_p = np.bincount(self.e_idx, weights=self.mass_e, minlength=num_p)

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.dirac_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dist += lr * dh[:, None]

            # check if converge
            max_change = np.max(dh / self.dirac_p)
            if max_change.size > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:.2f}%".format(i, max_change))

            if max_change <= 1:
                break
        # labels come from centroids
        if self.label_p is not None:
            self.e_predict = self.label_p[self.e_idx]

    def update_p(self, iter_p=0, reg_type=0, reg=0.01):
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

        num_p = self.data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = np.bincount(self.e_idx, minlength=num_p)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = np.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = np.max(np.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            self.data_p[:, i] = p_target

        # replace nan by original data
        mask = np.isnan(self.data_p).any(axis=1)
        self.data_p[mask] = self.data_p_original[mask].copy()
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < 0.01 else False

    def update_p_reg_potential(self, iter_p, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, label=None, reg=0.01):
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
            for l in np.unique(label):
                p_sub = p[label == l, :]
                # pairwise distance with smaller memory burden
                # |pi - pj|^2 = pi^2 + pj^2 - 2*pi*pj
                reg_term += np.sum((p_sub ** 2).sum(axis=1, keepdims=True) +
                                   (p_sub ** 2).sum(axis=1) -
                                   2 * p_sub.dot(p_sub.T))

            return np.sum((p - p0) ** 2.0) + reg * reg_term

        if np.unique(self.label_p).size == 1:
            warnings.warn("All known samples belong to the same class")

        p0 = np.zeros_like(self.data_p)
        num_p = self.data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = np.bincount(self.e_idx)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Abort this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        for i in range(p0.shape[1]):
            # update p to the centroid of their correspondences
            p_target = np.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = np.amax(np.abs((self.data_p[:, i] - p_target) / self.data_p[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        # regularize
        res = minimize(f, self.data_p, method='BFGS', args=(p0, self.label_p, reg))
        self.data_p = res.x.reshape(p0.shape)
        # return max change
        return True if max_change_pct < 0.01 else False

    def update_p_reg_transform(self, iter_p, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by an affine transformation
            which is estimated from the OT map.

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        assert self.data_p.shape[1] == 2, "dim has to be 2 for geometric transformation"

        p0 = np.zeros(self.data_p.shape)
        num_p = self.data_p.shape[0]
        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = np.bincount(self.e_idx, minlength=num_p)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Aborting this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(p0.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = np.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = np.max(np.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i]) + eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        pa = np.zeros(p0.shape)

        for idx, l in np.ndenumerate(np.unique(self.label_p)):
            idx_p_label = self.label_p == l
            p_sub = self.data_p[idx_p_label, :]
            p0_sub = p0[idx_p_label, :]
            # TODO estimating a high-dimensional transformation?
            T = tf.EuclideanTransform()
            # T = tf.AffineTransform()
            # T = tf.ProjectiveTransform()
            T.estimate(p_sub, p0_sub)
            pa[idx_p_label, :] = T(p_sub)

        pt = self.data_p.copy()
        T = tf.EuclideanTransform()
        T.estimate(pt, p0)
        pt = T(pt)

        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return max change
        return True if max_change_pct < 0.01 else False


class VotAP:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the area samples

    def __init__(self, data, sampling='square', label=None, mass_p=None, thres=1e-5, ratio=100, verbose=False):
        """ set up parameters
        Args:
            thres float: threshold to break loops
            ratio float: the ratio of num of e to the num of p
            data numpy ndarray: initial coordinates of p
            label numpy ndarray: labels of p
            mass_p numpy ndarray: weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            verbose   bool: console output verbose flag
            data_p    numpy ndarray: coordinates of p
            label_p   numpy ndarray: labels of p
            mass_p    numpy ndarray: mass of clusters of p
            dirac_p   numpy ndarray: dirac measure of p
        """

        if not isinstance(data, np.ndarray):
            raise Exception('input is not a numpy ndarray')
        self.data_p = data
        self.data_p_original = self.data_p.copy()
        num_p = data.shape[0]

        if label is not None and not isinstance(label, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')
        self.label_p = label

        if mass_p is not None and not isinstance(mass_p, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')
        if mass_p:
            self.dirac_p = mass_p
        else:
            self.dirac_p = np.ones(num_p) / num_p

        self.thres = thres
        self.verbose = verbose

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "dirac_p"
        self.mass_p = np.zeros(num_p)

        assert np.max(self.data_p) <= 1 and np.min(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

        num_p = self.data_p.shape[0]
        num_e = int(ratio * num_p)
        dim = self.data_p.shape[1]
        self.data_e, _ = utils.random_sample(num_e, dim, sampling=sampling)

        self.dist = cdist(self.data_p, self.data_e)**2
        self.e_idx = np.argmin(self.dist, axis=0)

    def map(self, plot_filename=None, beta=0.9, max_iter=1000, lr=0.2, lr_decay=100):
        """ map p into the area

        Args:
            sampling string: sampling area
            plot_filename string: filename of the gif image
            beta float: gradient descent momentum
            max_iter int: maximum number of iteration
            lr float: learning rate
            lr_decay float: learning rate decay

        :return:
        """

        num_p = self.data_p.shape[0]
        num_e = self.data_e.shape[0]

        imgs = []
        dh = 0

        for i in range(max_iter):
            # find nearest p for each e
            self.e_idx = np.argmin(self.dist, axis=0)

            # calculate total mass of each cell
            self.mass_p = np.bincount(self.e_idx, minlength=num_p) / num_e
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.dirac_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            self.dist += lr * dh[:, None]

            # check if converge
            max_change = np.max(dh / self.dirac_p)
            if max_change.size > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:.2f}%".format(i, max_change))
            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename and i % 10 == 0:
                fig = utils.plot_map(self.data_e, self.e_idx / (num_p - 1))
                img = utils.fig2data(fig)
                imgs.append(img)
            if max_change <= 1:
                break
        if plot_filename and imgs:
            imageio.mimsave(plot_filename, imgs, fps=4)
        # labels come from centroids
        if self.label_p is not None:
            self.label_e = self.label_p[self.e_idx]

        # update coordinates of p
        bincount = np.bincount(self.e_idx, minlength=num_p)
        if 0 in bincount:
            print('Empty cluster found, optimal transport did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = np.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
