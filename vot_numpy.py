# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Last modified: July 29th 2019


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
                 weight_p=None, weight_e=None, thres=1e-3, verbose=True):
        """ set up parameters

        p are centroids or source samples
        e are empirical or target samples
        In some literature, definitions of source and target are swapped.

        Throughout PyVot, the term "weight" is referred to the pre-defined value
        for each sample; the term "mass" of a p sample is referred to the weighted summation of
        all the e samples that are indexed to that p

        Args:
            data_p (numpy ndarray): coordinates of p
            data_e (numpy ndarray): coordinates of e
            label_p (numpy ndarray): labels of p
            label_e (numpy ndarray): labels of e
            weight_p (numpy ndarray): weights of p
            weight_e (numpy ndarray): weights of e
            thres (float): threshold to break loops
            verbose (bool): console output verbose flag

        Atts:
            data_p (numpy ndarray): coordinates of p
            data_e (numpy ndarray): coordinates of e
            label_p (numpy ndarray): labels of p
            label_e (numpy ndarray): labels of e
            weight_p (numpy ndarray): weight of p
            weight_e (numpy ndarray): weight of e
            mass_p (numpy ndarray): mass of p
            thres    (float): Threshold to break loops
            verbose   (bool): console output verbose flag
        """

        if not isinstance(data_p, np.ndarray):
            raise Exception('data_p is not a numpy ndarray')
        if not isinstance(data_e, np.ndarray):
            raise Exception('data_e is not a numpy ndarray')

        if label_p is not None and not isinstance(label_p, np.ndarray):
            raise Exception('label_p is not a numpy ndarray')
        if label_e is not None and not isinstance(label_e, np.ndarray):
            raise Exception('label_e is not a numpy ndarray')

        if weight_p is not None and not isinstance(weight_p, np.ndarray):
            raise Exception('weight_p is not a numpy ndarray')
        if weight_e is not None and not isinstance(weight_e, np.ndarray):
            raise Exception('weight_e is not a numpy ndarray')

        # deep copy all the data?
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.copy()

        num_p = data_p.shape[0]
        num_e = data_e.shape[0]

        self.label_p = label_p
        self.label_e = label_e

        self.thres = thres
        self.verbose = verbose

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "weight_p"
        self.mass_p = np.zeros(num_p)

        self.weight_p = weight_p if weight_p is not None else np.ones(num_p) / num_p
        self.weight_e = weight_e if weight_e is not None else np.ones(num_e) / num_e

        utils.assert_boundary(self.data_p)
        utils.assert_boundary(self.data_e)

    def cluster(self, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            e_idx (numpy ndarray): assignment of e to p
            pred_label_e (numpy ndarray): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """
        e_idx, pred_label_e = None, None
        for iter_p in range(max_iter_p):
            dist = cdist(self.data_p, self.data_e) ** 2
            e_idx, pred_label_e = self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay, early_stop=early_stop)
            if self.update_p(e_idx, iter_p):
                break
        return e_idx, pred_label_e

    def update_map(self, dist, max_iter=3000, lr=0.5, beta=0.9, lr_decay=200, early_stop=200):
        """ update assignment of each e as the map to p

        Args:
            dist (numpy ndarray): dist matrix across p and e
            max_iter   (int): max num of iterations
            lr       (float): gradient descent learning rate
            beta     (float): GD momentum
            lr_decay (int): learning rate decay frequency
            early_stop (int): early_stop check frequency

        Returns:
            e_idx (numpy ndarray): assignment of e to p
            pred_label_e (numpy ndarray): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = np.argmin(dist, axis=0)
            self.mass_p = np.bincount(e_idx, weights=self.weight_e, minlength=num_p)
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dist += lr * dh[:, None]

            # check if converge
            max_change = np.max((self.mass_p - self.weight_p)/self.weight_p)
            if max_change.size > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 10 == 0) or i % 100 == 0):
                print("{0:d}: mass diff {1:.2f}%".format(i, max_change))

            if max_change < 1:
                if self.verbose:
                    print("{0:d}: mass diff {1:.2f}%".format(i, max_change))
                break

            # early stop if loss does not decrease TODO better way to early stop?
            if early_stop >= 1:
                running_median.append(max_change)
                if len(running_median) >= early_stop:
                    if previous_median != 0 and\
                            np.abs(np.median(np.asarray(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = np.median(np.asarray(running_median))
                        running_median = []

        # labels come from p
        pred_label_e = self.label_p[e_idx] if self.label_p is not None else None

        return e_idx, pred_label_e

    @staticmethod
    def update_p_base(e_idx, data_p, data_e):
        """ base function to update each p to the centroids of its cluster

        Args:
            e_idx (numpy ndarray): assignment of e to p
            data_p (numpy ndarray): cluster centroids, p
            data_e (numpy ndarray): empirical samples, e
            p0 (numpy ndarray): iteration index

        Returns:
            p0 (numpy ndarray): new p
            max_change_pct (float): max_change
        """
        p0 = np.zeros_like(data_p)
        num_p = data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = np.bincount(e_idx, minlength=num_p)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = np.bincount(e_idx, weights=data_e[:, i], minlength=num_p) / bincount
            change_pct = np.max(np.abs((data_p[:, i] - p_target) / (data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target

        # replace nan by original data TODO replace nan by nn barycenter?
        mask = np.isnan(p0).any(axis=1)
        p0[mask] = data_p[mask].copy()

        return p0, max_change_pct

    def update_p(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (numpy ndarray): assignment of e to p
            iter_p (int): iteration index

        Returns:
            (bool): convergence or not, determined by max p change
        """

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)

        self.data_p = p0
        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # return convergence or not
        return True if max_change_pct < self.thres else False


class VotReg(Vot):
    """ variational optimal transportation with regularization on sample supports"""

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 weight_p=None, weight_e=None, thres=1e-3, verbose=True):
        super(VotReg, self).__init__(data_p, data_e, label_p=label_p, label_e=label_e,
                                     weight_p=weight_p, weight_e=weight_e, thres=thres, verbose=verbose)

    def cluster(self, reg_type=0, reg=0.01, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg      (float): regularization weight
            lr       (float): GD learning rate
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr_decay   (int): learning rate decay interval

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """
        e_idx, pred_label_e = None, None
        for iter_p in range(max_iter_p):
            dist = cdist(self.data_p, self.data_e) ** 2
            e_idx, pred_label_e = self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay, early_stop=early_stop)
            if self.update_p(e_idx, iter_p, reg_type, reg):
                break
        return e_idx, pred_label_e

    def update_p(self, e_idx, iter_p=0, reg_type=0, reg=0.01):
        """ update p

        Args:
            e_idx (numpy ndarray): assignment of e to p
            iter_p (int): iteration index
            reg_type (int or string): regularization type
            reg (float): regularizer weight

        Returns:
            bool: convergence or not
        """

        if reg_type == 1 or reg_type == 'potential':
            return self.update_p_reg_potential(e_idx, iter_p, reg)
        elif reg_type == 2 or reg_type == 'transform':
            return self.update_p_reg_transform(e_idx, iter_p, reg)
        else:
            return self.update_p_noreg(e_idx, iter_p)

    def update_p_noreg(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (numpy ndarray): assignment of e to p
            iter_p (int): iteration index

        Returns:
            bool: convergence or not, determined by max p change
        """

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)
        self.data_p = p0
        # replace nan by original data
        mask = np.isnan(self.data_p).any(axis=1)
        self.data_p[mask] = self.data_p_original[mask].copy()
        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False

    def update_p_reg_potential(self, e_idx, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            e_idx (numpy ndarray): assignment of e to p
            iter_p (int): index of the iteration of updating p
            reg (float): regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, label=None, reg=0.01):
            """ objective function incorporating labels

            Args:
                p  (numpy ndarray): p
                p0 (numpy ndarray): centroids of e
                label (numpy ndarray): labels of p
                reg (float): regularizer weight

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

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # regularize
        res = minimize(f, self.data_p, method='BFGS', args=(p0, self.label_p, reg))
        self.data_p = res.x.reshape(p0.shape)
        # return convergence or not
        return True if max_change_pct < self.thres else False

    def update_p_reg_transform(self, e_idx, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by an affine transformation
            which is estimated from the OT map.

        Args:
            e_idx (numpy ndarray): assignment of e to p
            iter_p (int): index of the iteration of updating p
            reg (float): regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        assert self.data_p.shape[1] == 2, "dim has to be 2 for geometric transformation"

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        pt = self.data_p.copy()
        pt = utils.estimate_transform_target(pt, p0)

        # regularize within each label
        # pt = np.zeros(p0.shape)
        # for label in np.unique(self.label_p):
        #     idx_p_label = self.label_p == label
        #     p_sub = self.data_p[idx_p_label, :]
        #     p0_sub = p0[idx_p_label, :]
        #     T = tf.EuclideanTransform()
        #     # T = tf.AffineTransform()
        #     # T = tf.ProjectiveTransform()
        #     T.estimate(p_sub, p0_sub)
        #     pt[idx_p_label, :] = T(p_sub)
        #
        # pt = self.data_p.copy()
        # T = tf.EuclideanTransform()
        # T.estimate(pt, p0)
        # pt = T(pt)

        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return convergence
        return True if max_change_pct < self.thres else False


class VotAP:
    """ Area Preserving by variational optimal transportation """
    # p are the centroids
    # e are the area samples
    # this is a separate class for area-preserving maps

    def __init__(self, data, sampling='square', label=None, mass_p=None, thres=1e-5, ratio=100, verbose=False):
        """ set up parameters
        Args:
            data (numpy ndarray): initial coordinates of p
            sampling (string): sampling area shape
            label (numpy ndarray): labels of p
            mass_p (numpy ndarray): weights of p
            thres (float): threshold to break loops
            ratio (float): the ratio of num of e to the num of p
            verbose (bool): boolean flag for verbose console output

        Atts:
            thres (float): Threshold to break loops
            lr    (float): Learning rate
            verbose (bool): console output verbose flag
            data_p  (numpy ndarray): coordinates of p
            label_p (numpy ndarray): labels of p
            mass_p  (numpy ndarray): mass of clusters of p
            weight_p (numpy ndarray): weight of p
        """

        if not isinstance(data, np.ndarray):
            raise Exception('input is not a numpy ndarray')

        if label is not None and not isinstance(label, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')

        if mass_p is not None and not isinstance(mass_p, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')

        self.data_p = data
        self.data_p_original = self.data_p.copy()
        num_p = self.data_p.shape[0]

        self.label_p = label
        if mass_p:
            self.weight_p = mass_p
        else:
            self.weight_p = np.ones(num_p) / num_p

        self.thres = thres
        self.verbose = verbose

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "weight_p"
        self.mass_p = np.zeros(num_p)

        utils.assert_boundary(self.data_p)

        num_e = int(ratio * num_p)
        dim = self.data_p.shape[1]
        self.data_e, _ = utils.random_sample(num_e, dim, sampling=sampling)

        self.dist = cdist(self.data_p, self.data_e)**2

    def map(self, plot_filename=None, beta=0.9, max_iter=1000, lr=0.5, lr_decay=200, early_stop=100):
        """ map p into the area

        Args:
            plot_filename (string): filename of the gif image
            beta (float): gradient descent momentum
            max_iter (int): maximum number of iteration
            lr (float): learning rate
            lr_decay (int): learning rate decay interval
            early_stop (int): early_stop checking frequency

        :return:
            e_idx (numpy ndarray): assignment of e to p
            pred_label_e (numpy ndarray): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        num_e = self.data_e.shape[0]

        imgs = []
        dh = 0

        e_idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e
            e_idx = np.argmin(self.dist, axis=0)

            # calculate total mass of each cell
            self.mass_p = np.bincount(e_idx, minlength=num_p) / num_e
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            self.dist += lr * dh[:, None]

            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename and i % 10 == 0:
                fig = utils.plot_map(self.data_e, e_idx / (num_p - 1))
                img = utils.fig2data(fig)
                imgs.append(img)

            # check if converge
            max_change = np.max((self.mass_p - self.weight_p) / self.weight_p)
            if max_change.size > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 10 == 0) or i % 100 == 0):
                print("{0:d}: mass diff {1:.2f}%".format(i, max_change))

            if max_change < 1:
                if self.verbose:
                    print("{0:d}: mass diff {1:.2f}%".format(i, max_change))
                break

            if early_stop > 0:
                # early stop if loss does not decrease TODO better way to early stop?
                running_median.append(max_change)
                if len(running_median) >= early_stop:
                    if previous_median != 0 and \
                            np.abs(np.median(np.asarray(running_median)) - previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = np.median(np.asarray(running_median))
                        running_median = []

            if max_change <= 1:
                break
        if plot_filename and imgs:
            imageio.mimsave(plot_filename, imgs, fps=4)
        # labels come from p
        pred_label_e = self.label_p[e_idx] if self.label_p is not None else None

        # update coordinates of p
        bincount = np.bincount(e_idx, minlength=num_p)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = np.bincount(e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount

        return e_idx, pred_label_e
