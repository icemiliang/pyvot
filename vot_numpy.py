# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 25th 2020
# Licence: MIT

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import imageio
import warnings
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
            y (numpy ndarray): coordinates of p
            x (numpy ndarray): coordinates of e
            label_y (numpy ndarray): labels of p
            label_e (numpy ndarray): labels of e
            weight_p (numpy ndarray): weight of p
            weight_e (numpy ndarray): weight of e
            mass_p (numpy ndarray): mass of p
            thres    (float): Threshold to break loops
            verbose   (bool): console output verbose flag
        """

        if not isinstance(data_p, np.ndarray):
            raise Exception('y is not a numpy ndarray')
        if not isinstance(data_e, np.ndarray):
            raise Exception('x is not a numpy ndarray')

        if label_p is not None and not isinstance(label_p, np.ndarray):
            raise Exception('label_y is not a numpy ndarray')
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

        self.weight_p = weight_p if weight_p is not None else np.ones(num_p) / num_p
        self.weight_e = weight_e if weight_e is not None else np.ones(num_e) / num_e

        utils.assert_boundary(self.data_p)
        utils.assert_boundary(self.data_e)

    def cluster(self, lr=0.5, max_iter_p=10, max_iter_h=5000, lr_decay=500, early_stop=-1):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            idx (numpy ndarray): assignment of e to p
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
            idx (numpy ndarray): assignment of e to p
            pred_label_e (numpy ndarray): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = np.argmin(dist, axis=0)
            mass_p = np.bincount(e_idx, weights=self.weight_e, minlength=num_p)
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dist += lr * dh[:, None]

            # check if converge
            max_change = np.max((mass_p - self.weight_p)/self.weight_p)
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
            p_target = np.bincount(e_idx, weights=data_e[:, i], minlength=num_p) / (bincount+eps)
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
        # for label in np.unique(self.label_y):
        #     idx_p_label = self.label_y == label
        #     p_sub = self.y[idx_p_label, :]
        #     p0_sub = p0[idx_p_label, :]
        #     T = tf.EuclideanTransform()
        #     # T = tf.AffineTransform()
        #     # T = tf.ProjectiveTransform()
        #     T.estimate(p_sub, p0_sub)
        #     pt[idx_p_label, :] = T(p_sub)
        #
        # pt = self.y.copy()
        # T = tf.EuclideanTransform()
        # T.estimate(pt, p0)
        # pt = T(pt)

        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return convergence
        return True if max_change_pct < self.thres else False


class VOTAP:
    """
        y are the centroids
        x are the area samples
        This is a minimum class for area-preserving maps
    """

    def __init__(self, data, sampling='square', label=None, nu=None, thres=1e-5, ratio=100, verbose=False):
        """ set up parameters
        """

        if not isinstance(data, np.ndarray):
            raise Exception('input is not a numpy ndarray')

        if label is not None and not isinstance(label, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')

        if nu is not None and not isinstance(nu, np.ndarray):
            raise Exception('label is neither a numpy array not a numpy ndarray')

        self.y = data
        self.data_p_original = self.y.copy()
        self.K = self.y.shape[0]

        self.label_y = label
        self.weight_p = nu if nu is not None else np.ones(self.K) / self.K

        self.thres = thres
        self.verbose = verbose

        utils.assert_boundary(self.y)

        self.N0 = int(ratio * self.K)
        ndim = self.y.shape[1]
        self.x, _ = utils.random_sample(self.N0, ndim, sampling=sampling)

        self.dist = cdist(self.y, self.x, 'sqeuclidean')

    def map(self, plot_filename=None, beta=0.9, max_iter=1000, lr=0.5, lr_decay=200, early_stop=100):
        """ map y into the area
        """

        imgs = []
        dh = 0

        idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e
            idx = np.argmin(self.dist, axis=0)

            # calculate total mass of each cell
            mass_p = np.bincount(idx, minlength=self.K) / self.N0
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            self.dist += lr * dh[:, None]

            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename and i % 10 == 0:
                fig = utils.plot_map(self.x, idx / (self.K - 1))
                img = utils.fig2data(fig)
                imgs.append(img)

            # check if converge
            max_change = np.max((mass_p - self.weight_p) / self.weight_p)
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
        # labels come from y
        pred_label_x = self.label_y[idx] if self.label_y is not None else None

        # update coordinates of y
        bincount = np.bincount(idx, minlength=self.K)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.y.shape[1]):
            # update p to the centroid of their correspondences
            self.y[:, i] = np.bincount(idx, weights=self.x[:, i], minlength=self.K) / bincount

        return idx, pred_label_x


class VOT:
    def __init__(self, y, x, nu=None, mu=None, lam=None, label_y=None, label_x=None, tol=1e-4, verbose=True):

        # marginals (x, mu)
        # centroids (y, nu)

        if type(x) is np.ndarray:
            if x.ndim == 2:
                self.x = [x]
            elif x.ndim == 3:
                self.x = [x[i] for i in range(x.shape[0])]
        else:
            self.x = x

        self.y = y.copy()
        self.y_original = y

        self.K = y.shape[0]  # number of centroids
        self.n = y.shape[1]  # number of dimensions
        self.N = len(self.x)  # number of marginals

        self.tol = tol
        self.verbose = verbose

        self.lam = lam if lam is not None else np.ones(self.N) / self.N

        self.idx = []
        self.mu = []
        self.sum_mu = []
        if mu is not None:
            # copy mu
            if type(mu) is np.ndarray:
                self.mu = [mu]
            else:
                self.mu = mu
            for m in self.mu:
                self.idx.append(np.ones_like(m, dtype=np.int64))
                self.sum_mu.append(np.sum(m))
        else:
            # create uniform mu
            self.mu = []
            self.idx = []
            for i in range(self.N):
                N_i = self.x[i].shape[0]
                self.mu.append(1. / N_i)
                self.idx.append(np.zeros(N_i, dtype=np.int64))
                self.sum_mu.append(1.)

        if nu is not None:
            self.nu = nu
            self.sum_nu = np.sum(self.nu)
            if abs(self.sum_nu - 1) > 1e-3:
                self.nu /= self.sum_nu
                self.sum_nu = 1
                self.mu = [m / self.sum_nu for m in self.mu]
        else:
            self.nu = 1. / self.K
            self.sum_nu = 1.

        self.label_y = label_y
        self.label_x = []

        # all data should be in (-1, 1) in each dimension
        utils.assert_boundary(self.y)
        for i in range(self.N):
            utils.assert_boundary(self.x[i])

    def cluster(self, lr=0.5, max_iter_y=10, max_iter_h=3000, lr_decay=200, stop=-1, beta=0, reg=0., keep_idx=False, space='euclidean', icp=False):
        """ compute Wasserstein clustering
        """

        lrs = [lr / m for m in self.sum_mu]
        idxs = []
        for it in range(max_iter_y):
            for i in range(self.N):
                print("solving marginal #" + str(i))
                if space == 'spherical':
                    dist = np.matmul(self.y, self.x[i].T)
                else:
                    dist = cdist(self.y, self.x[i], 'sqeuclidean')
                output = self.update_map(i, dist, max_iter_h, lr=lrs[i], lr_decay=lr_decay, beta=beta, stop=stop, reg=reg, keep_idx=keep_idx, space=space)
                self.idx[i] = output['idx']
                if keep_idx:
                    idxs.append(output['idxs'])

            if icp:
                if self.update_x(it):
                    break
            elif self.update_y(it, space=space):
                break
        output = dict()
        output['idxs'] = idxs

        # pass label from y to x
        if self.label_y is not None:
            for i in range(self.N):
                self.label_x.append(self.label_y[self.idx[i]])

        # compute W_2^2
        twd = 0
        wds = []
        for i in range(self.N):
            tmp = (self.y[self.idx[i], :] - self.x[i]) ** 2
            wd = np.sum(np.sum(tmp, axis=1) * self.mu[i])
            twd += wd
            wds.append(wd)

        output['wd'] = twd
        output['wds'] = wds
        return output

    def update_map(self, i, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, stop=200, reg=0., keep_idx=False, space='euclidean'):
        """ update assignment of each e as the ot_map to y
        """

        dh = 0
        idx = None
        idxs = []
        running_median, previous_median = [], 0

        h = np.ones(self.K) if space == 'spherical' else None

        dist_original = 0 if reg == 0 else dist.copy()

        for it in range(max_iter):
            # find nearest y for each x and add mass to y
            if space == 'spherical':
                idx = np.argmin(dist / np.cos(h)[:, None], axis=0)
            else:
                idx = np.argmin(dist, axis=0)
            if keep_idx:
                idxs.append(idx)
            if isinstance(self.mu[i], float):
                mass = np.bincount(idx, minlength=self.K) * self.mu[i]
            else:
                mass = np.bincount(idx, weights=self.mu[i], minlength=self.K)

            # gradient descent with momentum and decay
            dh = beta * dh + (1 - beta) * (mass - self.nu)
            if it != 0 and it % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dh *= lr
            if space == 'spherical':
                h += dh
            else:
                dist += dh[:, None]

            # check if converge
            if self.verbose and it % 1000 == 0:
                print(dh)
            max_change = np.max((mass - self.nu) / self.nu)
            if max_change.size > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 20 and i % 1 == 0) or i % 200 == 0):
                print("{0:d}: mass diff {1:.2f}%".format(it, max_change))

            if max_change < 1:
                if self.verbose:
                    print("{0:d}: mass diff {1:.2f}%".format(it, max_change))
                break

            # early stop if loss does not decrease TODO better way to early stop?
            if stop >= 1:
                running_median.append(max_change)
                if len(running_median) >= stop:
                    if previous_median != 0 and\
                            np.abs(np.median(np.array(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = np.median(np.array(running_median))
                        running_median = []

        if reg != 0.:
            idx = np.argmin(reg / (1 + reg) * dist + 1 / (1 + reg) * dist_original, axis=0)

        output = dict()
        output['idx'] = idx
        output['idxs'] = idxs
        return output

    @staticmethod
    def update_y_base(idx, y, x):
        """ base function to update each y to the centroids of its cluster
        """

        new_y = np.zeros_like(y)
        max_change_pct = 0.0
        K, ndim = y.shape

        bincount = np.bincount(idx, minlength=K)
        if 0 in bincount:
            print('Empty cluster found, OT probably did not converge\n'
                  'Try a different lr or max_iter assuming the input is correct.')
            # return False
        eps = 1e-8

        # update y to the centroid of their correspondences one dimension at a time
        # for spherical domains, use Euclidean barycenter to approximate and project it to the surface
        for n in range(ndim):
            mass_center = np.bincount(idx, weights=x[:, n], minlength=K) / (bincount + eps)
            change_pct = np.max(np.abs((y[:, n] - mass_center) / (y[:, n]) + eps))
            max_change_pct = max(max_change_pct, change_pct)
            new_y[:, n] = mass_center

        # replace nan by original data TODO replace nan by nn barycenter?
        mask = np.isnan(new_y).any(axis=1)
        new_y[mask] = y[mask].copy()

        return new_y, max_change_pct

    def update_y(self, it=0, idx=None, space='euclidean', icp=False):
        """ update each y to the centroids of its cluster
        """
        if idx is None:
            idx = self.idx
        max_change_pct = 1e9

        y = np.zeros((self.N, self.K, self.n))
        if icp:
            yR = np.zeros((self.N, self.K, self.n))
        for i in range(self.N):
            y[i], change = self.update_y_base(idx[i], self.y, self.x[i])
            max_change_pct = max(max_change_pct, change)
            if icp:
                yR[i] = utils.estimate_transform_target(self.y, y[i])

        if icp:
            y = yR

        self.y = np.sum(y * self.lam[:, None, None], axis=0)

        if space == 'spherical':
            self.y /= np.linalg.norm(self.y, axis=1, keepdims=True)

        if self.verbose:
            print("iter {0:d}: max centroid change {1:.2f}%".format(it, 100 * max_change_pct))

        return True if max_change_pct < self.tol else False

    def update_x(self, it=0, idx=None):
        """ update each x
        """
        if idx is None:
            idx = self.idx
        max_change_pct = 1e9

        y = np.zeros((self.N, self.K, self.n))
        for i in range(self.N):
            y[i], change = self.update_y_base(idx[i], self.y, self.x[i])
            max_change_pct = max(max_change_pct, change)

        self.y = np.sum(y * self.lam[:, None, None], axis=0)

        for i in range(self.N):
            r, t = utils.estimate_inverse_transform(self.y, y[i])
            self.x[i] = (np.matmul(r, self.x[i].T) + t).T

        if self.verbose:
            print("iter {0:d}: max centroid change {1:.2f}%".format(it, 100 * max_change_pct))

        return True if max_change_pct < self.tol else False


class VOTREG(VOT):
    """ variational optimal transportation with regularization on sample supports"""

    def map(self, reg_type, reg, lr=0.5, max_iter_y=10, max_iter_h=3000, lr_decay=200, stop=-1, keep_idx=False):
        """ compute Wasserstein clustering
        """
        lrs = [lr / m for m in self.sum_mu]
        idxs = []
        for iter_y in range(max_iter_y):
            dist = cdist(self.y, self.x[0], 'sqeuclidean')
            output = self.update_map(0, dist, max_iter_h, lr=lrs[0], lr_decay=lr_decay, stop=stop)
            self.idx[0] = output['idx']
            if keep_idx:
                idxs.append(output['idxs'])
            if reg_type == 1 or reg_type == 'potential':
                if self.update_y_potential(iter_y, reg):
                    break
            elif reg_type == 2 or reg_type == 'transform':
                if self.update_p_transform(iter_y, reg):
                    break
            else:
                raise Exception('regularization type not defined')

        # pass label from y to x
        if self.label_y is not None:
            for i in range(self.N):
                self.label_x.append(self.label_y[self.idx[i]])

        output = dict()
        output['idxs'] = idxs
        return output

    def update_y_potential(self, iter_y=0, reg=0.01):
        """ update each p to the centroids of its cluster,
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

        if np.unique(self.label_y).size == 1:
            warnings.warn("All known samples belong to the same class")

        y0, max_change_pct = self.update_y_base(self.idx[0], self.y, self.x[0])

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_y, max_change_pct))

        # regularize
        res = minimize(f, self.y, method='BFGS', args=(y0, self.label_y, reg))
        self.y = res.x.reshape(y0.shape)

        return True if max_change_pct < self.tol else False

    def update_p_transform(self, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster,
        """

        assert self.y.shape[1] == 2, "dim has to be 2 for geometric transformation"

        p0, max_change_pct = self.update_y_base(self.idx[0], self.y, self.x[0])

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        pt = np.zeros_like(self.y)
        pt = utils.estimate_transform_target(pt, p0)

        # regularize within each label
        # pt = np.zeros(p0.shape)
        # for label in np.unique(self.label_y):
        #     idx_p_label = self.label_y == label
        #     p_sub = self.y[idx_p_label, :]
        #     p0_sub = p0[idx_p_label, :]
        #     T = tf.EuclideanTransform()
        #     # T = tf.AffineTransform()
        #     # T = tf.ProjectiveTransform()
        #     T.estimate(p_sub, p0_sub)
        #     pt[idx_p_label, :] = T(p_sub)
        #
        # pt = self.y.copy()
        # T = tf.EuclideanTransform()
        # T.estimate(pt, p0)
        # pt = T(pt)

        self.y = pt
        # self.y = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return convergence
        return True if max_change_pct < self.tol else False
