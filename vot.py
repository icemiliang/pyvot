# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 15th 2019


import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from skimage import transform as tf
import imageio
import utils


class Vot:
    """ variational optimal transportation """

    def import_data_from_file(self, pfilename, efilename, mass=False, label=True):
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

        p_data = np.loadtxt(pfilename, delimiter=",")
        e_data = np.loadtxt(efilename, delimiter=",")

        if label and mass:
            self.import_data(p_data[:, 2:], e_data[:, 2:],
                             yp=p_data[:, 0], ye=e_data[:, 0],
                             mass_p=p_data[:, 1], mass_e=e_data[:, 0])
        elif label and not mass:
            self.import_data(p_data[:, 1:], e_data[:, 1:],
                             yp=p_data[:, 0], ye=e_data[:, 0])
        elif not label and mass:
            self.import_data(p_data[:, 1:], e_data[:, 1:],
                             mass_p=p_data[:, 0], mass_e=e_data[:, 0])
        else:
            self.import_data(p_data, e_data)

    def import_data(self, Xp, Xe, yp=None, ye=None, mass_p=None, mass_e=None):
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
        self.p_coor_original = np.copy(Xp)

        # "p_mass" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.p_mass = np.zeros(self.num_p)

        if abs(np.sum(self.p_dirac) - np.sum(self.e_mass)) > 1e-6:
            warnings.warn("Total mass of e does not equal to total mass of p")

    def cluster(self, reg_type=0, reg=0.01, lr=0.2, max_iter_p=10, max_iter_h=2000):
        """ compute Wasserstein clustering

        Args:
            reg int: flag for regularization, 0 means no regularization

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        for iter_p in range(max_iter_p):
            cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            self.update_map(cost_base, max_iter_h, lr=lr)
            if self.update_p(iter_p, reg_type, reg):
                break

    def update_map(self, cost_base, max_iter, lr=0.2, beta=0.9, lr_decay=50):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            bool: convergence or not, determined by max derivative change
        """

        h = np.zeros(self.num_p)
        for i in range(max_iter):
            # update dist matrix
            cost = cost_base - h[:, np.newaxis]
            # find nearest p for each e
            self.e_idx = np.argmin(cost, axis=0)
            # labels come from centroids
            self.e_predict = self.p_label[self.e_idx]
            self.p_mass = np.bincount(self.e_idx, weights=self.e_mass, minlength=self.num_p)
            # update gradient and h
            dh = self.p_mass - self.p_dirac

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.p_mass - self.p_dirac)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            h -= lr * dh

            # check if converge and return max derivative
            index = np.argmax(dh)
            if isinstance(index, np.ndarray):
                index = index[0]
            max_change = dh[index]
            max_change_pct = max_change * 100 / self.p_mass[index]
            if max_change_pct <= 1:
                break

    def update_p(self, iter_p, reg_type=0, reg=0.01):
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

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = np.bincount(self.e_idx)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        for i in range(self.p_coor.shape[1]):
            # update p to the centroid of their correspondences
            p_target = np.bincount(self.e_idx, weights=self.e_coor[:, i], minlength=self.num_p) / bincount
            change_pct = np.amax(np.abs((self.p_coor[:, i] - p_target)/self.p_coor[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            self.p_coor[:, i] = p_target
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
            for idx, l in np.ndenumerate(np.unique(label)):
                p_sub = p[label == l, :]
                # pairwise distance with smaller memory burden
                # |pi - pj|^2 = pi^2 + pj^2 - 2*pi*pj
                reg_term += np.sum((p_sub ** 2).sum(axis=1, keepdims=True) +
                                   (p_sub ** 2).sum(axis=1) -
                                   2 * p_sub.dot(p_sub.T))

            return np.sum((p - p0) ** 2.0) + reg * reg_term

        if np.unique(self.p_label).size == 1:
            warnings.warn("All known samples belong to the same class")

        p0 = np.zeros_like(self.p_coor)

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
            p_target = np.bincount(self.e_idx, weights=self.e_coor[:, i], minlength=self.num_p) / bincount
            change_pct = np.amax(np.abs((self.p_coor[:, i] - p_target)/self.p_coor[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        # regularize
        res = minimize(f, self.p_coor, method='BFGS', args=(p0, self.p_label, reg))
        self.p_coor = res.x.reshape(p0.shape)
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

        def f(p, p0, pa, reg=0.01):
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

        assert self.p_coor.shape[1] == 2, "dim has to be 2 for geometric transformation"

        p0 = np.zeros_like(self.p_coor)
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
            p_target = np.bincount(self.e_idx, weights=self.e_coor[:, i], minlength=self.num_p) / bincount
            change_pct = np.amax(np.abs((self.p_coor[:, i] - p_target)/self.p_coor[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        pa = np.zeros(p0.shape)

        for idx, l in np.ndenumerate(np.unique(self.p_label)):
            idx_p_label = self.p_label == l
            p_sub = self.p_coor[idx_p_label, :]
            p0_sub = p0[idx_p_label, :]
            # TODO estimating a high-dimensional transformation?
            T = tf.EuclideanTransform()
            # T = tf.AffineTransform()
            # T = tf.ProjectiveTransform()
            T.estimate(p_sub, p0_sub)
            pa[idx_p_label, :] = T(p_sub)

        res = minimize(f, self.p_coor, method='BFGS', args=(p0, pa, reg))
        self.p_coor = res.x.reshape(p0.shape)
        # return max change
        return True if max_change_pct < 0.01 else False


class VotAP:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the area samples

    def __init__(self, data, label=None, mass_p=None, thres=1e-5, ratio=100, verbose=True):
        """ set up parameters
        Args:
            thres float: threshold to break loops
            ratio float: the ratio of num of e to the num of p
            data np.ndarray(np,dim+): initial coordinates of p
            label np.ndarray(num_p,): labels of p
            mass_p np.ndarray(num_p,): weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            ratio    float: ratio of num_e to num_p
            verbose   bool: console output verbose flag
            num_p      int: number of p
            X_p    numpy ndarray: coordinates of p
            y_p    numpy ndarray: labels of p
            mass_p numpy ndarray: mass of clusters of p

        """
        self.thres = thres
        self.verbose = verbose
        self.p_dirac = None
        self.ratio = ratio

        self.has_mass = mass_p is not None
        self.has_label = label is not None

        num_p = data.shape[0]
        self.label_p = label.astype(int) if not label is None else -np.ones(num_p).astype(int)
        self.p_dirac = mass_p if not mass_p is None else np.ones(num_p) / num_p
        self.data_p = data
        self.data_p_original = data.copy()
        # "mass_p" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.mass_p = np.zeros(num_p)

        assert np.amax(self.data_p) <= 1 and np.amin(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

    def map(self, sampling='unisquare', plot_filename=None, beta=0.9, max_iter=1000, lr=0.2, lr_decay=50):
        """ map p into the area

        Args:
            sampling string: sampling area
            plot_filename string: filename of the gif image
            beta float: gradient descent momentum
            max_iter int: maximum number of iteration
            lr float: learning rate
            lr_decay float: learning rate decay

        Atts:
            num_p int: number of p
            num_e int: number of e
            dim int: dimentionality
            data_e numpy ndarray: coordinates of e
            label_e numpy ndarray: label of e
            base_dist numpy ndarray: pairwise distance between p and e
            h  numpy ndarray: VOT optimizer, "height vector
            dh  numpy ndarray: gradient of h
            max_change float: maximum gradient change
            max_change_pct float: relative maximum gradient change
            imgs list: list of plots to show mapping progress
            e_idx numpy ndarray: p index of every e

        :return:
        """
        num_p = self.data_p.shape[0]
        num_e = self.ratio * num_p
        dim = self.data_p.shape[1]
        self.data_e, self.label_e = utils.random_sample(num_e, dim, sampling=sampling)
        base_dist = cdist(self.data_p, self.data_e, 'sqeuclidean')
        self.e_idx = np.argmin(base_dist, axis=0)
        h = np.zeros(num_p)
        imgs = []
        dh = 0

        for i in range(max_iter):
            dist = base_dist - h[:, None]
            # find nearest p for each e
            self.e_idx = np.argmin(dist, axis=0)

            # calculate total mass of each cell
            self.mass_p = np.bincount(self.e_idx, minlength=num_p) / num_e

            # labels come from centroids
            if self.has_label:
                self.label_e = self.label_p[self.e_idx]

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.p_dirac)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            h -= lr * dh

            # check if converge and return max derivative
            index = np.argmax(dh)
            if isinstance(index, np.ndarray):
                index = index[0]
            max_change = dh[index]
            max_change_pct = max_change * 100 / self.mass_p[index]

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:g} ({2:.2f}%)".format(i, max_change, max_change_pct))
            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename:
                fig = utils.plot_map(self.data_e, self.e_idx / (num_p - 1))
                img = utils.fig2data(fig)
                imgs.append(img)
            if max_change_pct <= 1:
                break
        if plot_filename and imgs:
            imageio.mimsave(plot_filename, imgs, fps=4)

        # update coordinates of p
        bincount = np.bincount(self.e_idx)
        if 0 in bincount:
            print('Empty cluster found, optimal transport did not converge\nTry larger lr or max_iter')
            return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = np.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
