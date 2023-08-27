# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT

import torch
import torch.optim as optim
import imageio
import warnings
import utils_torch as utils


class VOTAP:
    """
        y are the centroids
        x are the area samples
        This is a minimum class for area-preserving maps
    """

    def __init__(self, data, sampling='square', label=None, nu=None, thres=1e-5, ratio=100, device='cpu', verbose=False):
        """ set up parameters
        """

        if not isinstance(data, torch.Tensor):
            raise Exception('input is not a torch tensor')

        if label is not None and not isinstance(label, torch.Tensor):
            raise Exception('label is neither a numpy array not a torch tensor')

        if nu is not None and not isinstance(nu, torch.Tensor):
            raise Exception('label is neither a numpy array not a torch tensor')

        self.y = data
        self.y_original = self.y.clone()
        self.K = self.y.shape[0]

        self.label_y = label
        self.weight_p = nu if nu is not None else torch.ones(self.K).double().to(device) / self.K

        self.thres = thres
        self.device = device
        self.verbose = verbose

        utils.assert_boundary(self.y)

        self.N0 = int(ratio * self.K)
        self.x, _ = utils.random_sample(self.N0, self.y.shape[1], sampling=sampling)

        self.dist = torch.cdist(self.y, self.x.double()).to(device) ** 2

    def map(self, plot_filename=None, beta=0.9, max_iter=1000, lr=0.5, lr_decay=200, early_stop=100):
        """ map y into the area
        """

        imgs = []
        dh = 0

        idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest y for each x
            idx = torch.argmin(self.dist, axis=0)

            # calculate total mass of each cell
            mass_p = torch.bincount(idx, minlength=self.K) / self.N0
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
            max_change = torch.max((mass_p - self.weight_p) / self.weight_p)
            if torch.numel(max_change) > 1:
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
                            torch.abs(torch.median(torch.tensor(running_median)) - previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.tensor(running_median))
                        running_median = []

            if max_change <= 1:
                break
        if plot_filename and len(imgs) > 0:
            imageio.mimsave(plot_filename, imgs, fps=4)
        # labels come from y
        pred_label_x = self.label_y[idx] if self.label_y is not None else None

        # update coordinates of y
        bincount = torch.bincount(idx, minlength=self.K)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.y.shape[1]):
            # update y to the centroid of their correspondences
            self.y[:, i] = torch.bincount(idx, weights=self.x[:, i], minlength=self.K) / bincount

        return idx, pred_label_x


class VOT:
    def __init__(self, y, x, nu=None, mu=None, lam=None, label_y=None, label_x=None, tol=1e-4, verbose=True, device='cpu'):

        # marginals (x, mu)
        # centroids (y, nu)

        if type(x) is torch.Tensor:
            if x.dim() == 2:
                self.x = [x]
            elif x.dim() == 3:
                self.x = [x[i] for i in range(x.shape[0])]
        else:
            self.x = x

        self.y = y.clone()
        self.y_original = y

        self.K = y.shape[0]  # number of centroids
        self.n = y.shape[1]  # number of dimensions
        self.N = len(self.x)  # number of marginals

        self.tol = tol
        self.verbose = verbose
        self.device = device

        self.lam = lam if lam is not None else torch.ones(self.N) / self.N

        self.idx = []
        self.mu = []
        self.sum_mu = []
        if mu is not None:
            # copy mu
            if type(mu) is torch.Tensor:
                self.mu = [mu]
            else:
                self.mu = mu
            for m in self.mu:
                self.idx.append(torch.ones_like(m, dtype=torch.int64))
                self.sum_mu.append(torch.sum(m))
        else:
            # create uniform mu
            self.mu = []
            self.idx = []
            for i in range(self.N):
                N_i = self.x[i].shape[0]
                self.mu.append(1. / N_i)
                self.idx.append(torch.zeros(N_i, dtype=torch.int64))
                self.sum_mu.append(1.)

        if nu is not None:
            self.nu = nu
            self.sum_nu = torch.sum(self.nu)
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
                    dist = torch.matmul(self.y, self.x[i].T)
                else:
                    dist = torch.cdist(self.y, self.x[i], p=2) ** 2
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
            wd = torch.sum(torch.sum(tmp, dim=1) * self.mu[i])
            twd += wd
            wds.append(wd)

        output['wd'] = twd
        output['wds'] = wds
        return output

    def update_map(self, i, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, stop=200, reg=0., keep_idx=False, space='euclidean'):
        """ update assignment of each x as the ot_map to y
        """

        dh = 0
        idx = None
        idxs = []
        running_median, previous_median = [], 0

        h = torch.ones(self.K) if space == 'spherical' else None

        dist_original = 0 if reg == 0 else dist.clone()

        for it in range(max_iter):
            # find nearest y for each x and add mass to y
            if space == 'spherical':
                idx = torch.argmin(dist / torch.cos(h)[:, None], dim=0)
            else:
                idx = torch.argmin(dist, dim=0)
            if keep_idx:
                idxs.append(idx)
            if isinstance(self.mu[i], float):
                mass = torch.bincount(idx, minlength=self.K) * self.mu[i]
            else:
                mass = torch.bincount(idx, weights=self.mu[i], minlength=self.K)

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
            max_change = torch.max((mass - self.nu) / self.nu)
            if torch.numel(max_change) > 1:
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
                            torch.abs(torch.median(torch.Tensor(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.Tensor(running_median))
                        running_median = []

        if reg != 0.:
            idx = torch.argmin(reg / (1 + reg) * dist + 1 / (1 + reg) * dist_original, dim=0)

        output = dict()
        output['idx'] = idx
        output['idxs'] = idxs
        return output

    @staticmethod
    def update_y_base(idx, y, x):
        """ base function to update each y to the centroids of its cluster
        """

        new_y = torch.zeros_like(y)
        max_change_pct = 0.0
        K, ndim = y.shape

        bincount = torch.bincount(idx, minlength=K)
        if 0 in bincount:
            print('Empty cluster found, OT probably did not converge\n'
                  'Try a different lr or max_iter assuming the input is correct.')
            # return False
        eps = 1e-8

        # update y to the centroid of their correspondences one dimension at a time
        # for spherical domains, use Euclidean barycenter to approximate and project it to the surface
        for n in range(ndim):
            mass_center = torch.bincount(idx, weights=x[:, n], minlength=K) / (bincount + eps)
            change_pct = torch.max(torch.abs((y[:, n] - mass_center) / (y[:, n]) + eps))
            max_change_pct = max(max_change_pct, change_pct)
            new_y[:, n] = mass_center

        # replace nan by original data TODO replace nan by nn barycenter?
        mask = torch.isnan(new_y).any(dim=1)
        new_y[mask] = y[mask].clone()

        return new_y, max_change_pct

    def update_y(self, it=0, idx=None, space='euclidean', icp=False):
        """ update each y to the centroids of its cluster
        """
        if idx is None:
            idx = self.idx
        max_change_pct = 1e9

        y = torch.zeros((self.N, self.K, self.n), dtype=torch.float64).to(self.device)
        if icp:
            yR = torch.zeros((self.N, self.K, self.n), dtype=torch.float64).to(self.device)
        for i in range(self.N):
            y[i], change = self.update_y_base(idx[i], self.y, self.x[i])
            max_change_pct = max(max_change_pct, change)
            if icp:
                yR[i] = utils.estimate_transform_target(self.y, y[i])

        if icp:
            y = yR

        self.y = torch.sum(y * self.lam[:, None, None], dim=0)

        if space == 'spherical':
            self.y /= torch.linalg.norm(self.y, axis=1, keepdims=True)

        if self.verbose:
            print("iter {0:d}: max centroid change {1:.2f}%".format(it, 100 * max_change_pct))

        return True if max_change_pct < self.tol else False

    def update_x(self, it=0, idx=None):
        """ update each x
        """
        if idx is None:
            idx = self.idx
        max_change_pct = 1e9

        y = torch.zeros((self.N, self.K, self.n), dtype=torch.float64).to(self.device)
        for i in range(self.N):
            y[i], change = self.update_y_base(idx[i], self.y, self.x[i])
            max_change_pct = max(max_change_pct, change)

        self.y = torch.sum(y * self.lam[:, None, None], dim=0)

        for i in range(self.N):
            r, t = utils.estimate_inverse_transform(self.y, y[i])
            self.x[i] = (torch.matmul(r, self.x[i].T) + t).T

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
            dist = torch.cdist(self.y, self.x[0], p=2) ** 2
            output = self.update_map(0, dist, max_iter_h, lr=lrs[0], lr_decay=lr_decay, stop=stop)
            self.idx[0] = output['idx']
            if keep_idx:
                idxs.append(output['idxs'])
            if reg_type == 1 or reg_type == 'potential':
                if self.update_y_potential(iter_y, reg):
                    break
            elif reg_type == 2 or reg_type == 'transform':
                if self.update_y_transform(iter_y, reg):
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

        def f(p, p0, label=None, reg=0.1):
            """ objective function incorporating labels

            Args:
                p  pytorch floattensor:   p
                p0 pytorch floattensor:  centroids of e
                label pytorch inttensor: labels of p
                reg float: regularizer weight

            Returns:
                pytorch inttensor: f = sum(|p-p0|^2) + reg * sum(1(li == lj)*|pi-pj|^2)
            """

            reg_term = 0.0
            for l in torch.unique(label):
                p_sub = p[label == l, :]
                reg_term += torch.pow(torch.pdist(p_sub, p=2), 2).sum()

            return torch.sum((p - p0) ** 2.0) + reg * reg_term

        if torch.unique(self.label_y).size == 1:
            warnings.warn("All known samples belong to the same class")

        y0, max_change_pct = self.update_y_base(self.idx[0], self.y.detach(), self.x)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_y, max_change_pct))

        # regularize
        optimizer = optim.SGD([self.y], lr=0.05)
        for _ in range(10):
            optimizer.zero_grad()
            loss = f(self.y, y0, self.label_y, reg=reg)
            loss.backward()
            optimizer.step()


        return True if max_change_pct < self.tol else False

    def update_y_transform(self, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by an affine transformation
            which is estimated from the OT ot_map.

        Args:
            e_idx (torch Tensor): assignment of e to p
            iter_p (int): index of the iteration of updating p
            reg (float): regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        assert self.y.shape[1] == 3 or self.y.shape[1] == 2, "dim has to be 2 or 3 for geometric transformation"

        p0, max_change_pct = self.update_y_base(self.idx[0], self.y.detach(), self.x)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # pt = utils.estimate_transform_target_pytorch(self.y.detach(), p0)
        pt = utils.estimate_transform_target(self.y.detach(), p0)
        # regularize within each label
        # pt = torchzeros(p0.shape)
        # for label in torchunique(self.label_y):
        #     idx_p_label = self.label_y == label
        #     p_sub = self.y[idx_p_label, :]
        #     p0_sub = p0[idx_p_label, :]
        #     T = tf.EuclideanTransform()
        #     # T = tf.AffineTransform()
        #     # T = tf.ProjectiveTransform()
        #     T.estimate(p_sub, p0_sub)
        #     pt[idx_p_label, :] = T(p_sub)
        #
        # pt = self.y.clone()
        # T = tf.EuclideanTransform()
        # T.estimate(pt, p0)
        # pt = T(pt)

        # pt = p0 + pt

        self.y = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return convergence
        return True if max_change_pct < self.tol else False
