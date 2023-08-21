# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT

import torch
import torch.optim as optim
import imageio
import warnings
import utils_torch as utils


class Vot:
    """ variational optimal transportation """

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 weight_p=None, weight_e=None, thres=1e-3, verbose=True, device='cpu'):
        """ set up parameters

        p are centroids or source samples
        e are empirical or target samples
        In some literature, definitions of source and target are swapped.

        Throughout PyVot, the term "weight" is referred to the pre-defined value
        for each sample; the term "mass" of a p sample is referred to the weighted summation of
        all the e samples that are indexed to that p

        Args:
            data_p (pytorch Tensor): coordinates of p
            data_e (pytorch Tensor): coordinates of e
            label_p (pytorch Tensor): labels of p
            label_e (pytorch Tensor): labels of e
            weight_p (pytorch Tensor): weights of p
            weight_e (pytorch Tensor): weights of e
            thres (float): threshold to break loops
            verbose (bool): console output verbose flag

        Atts:
            y (pytorch Tensor): coordinates of p
            x (pytorch Tensor): coordinates of e
            label_y (pytorch Tensor): labels of p
            label_e (pytorch Tensor): labels of e
            weight_p (pytorch Tensor): weight of p
            weight_e (pytorch Tensor): weight of e
            mass_p (pytorch Tensor): mass of p
            thres    (float): Threshold to break loops
            verbose   (bool): console output verbose flag
        """

        if not isinstance(data_p, torch.Tensor):
            raise Exception('y is not a pytorch Tensor')
        if not isinstance(data_e, torch.Tensor):
            raise Exception('x is not a pytorch Tensor')

        if label_p is not None and not isinstance(label_p, torch.Tensor):
            raise Exception('label_y is not a pytorch Tensor')
        if label_e is not None and not isinstance(label_e, torch.Tensor):
            raise Exception('label_e is not a pytorch Tensor')

        if weight_p is not None and not isinstance(weight_p, torch.Tensor):
            raise Exception('weight_p is not a pytorch Tensor')
        if weight_e is not None and not isinstance(weight_e, torch.Tensor):
            raise Exception('weight_e is not a pytorch Tensor')

        # deep copy all the data?
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.clone()

        num_p = data_p.shape[0]
        num_e = data_e.shape[0]

        self.label_p = label_p
        self.label_e = label_e

        self.thres = thres
        self.verbose = verbose
        self.device = device

        self.weight_p = weight_p if weight_p is not None else torch.ones(num_p).double() / num_p
        self.weight_e = weight_e if weight_e is not None else torch.ones(num_e).double() / num_e

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
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """
        e_idx, pred_label_e = None, None
        for iter_p in range(max_iter_p):
            dist = torch.cdist(self.data_p, self.data_e) ** 2
            e_idx, pred_label_e = self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay, early_stop=early_stop)
            if self.update_p(e_idx, iter_p):
                break
        return e_idx, pred_label_e

    def update_map(self, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, early_stop=200):
        """ update assignment of each e as the ot_map to p

        Args:
            dist (pytorch Tensor): dist matrix across p and e
            max_iter   (int): max num of iterations
            lr       (float): gradient descent learning rate
            beta     (float): GD momentum
            lr_decay (int): learning rate decay frequency
            early_stop (int): early_stop check frequency

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = torch.argmin(dist, dim=0)
            mass_p = torch.bincount(e_idx, weights=self.weight_e, minlength=num_p).double()
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dh *= lr
            dist += dh[:, None]

            # check if converge
            if self.verbose and i % 1000 == 0:
                print(dh)
            max_change = torch.max((mass_p - self.weight_p)/self.weight_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 1 == 0) or i % 100 == 0):
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
                            torch.abs(torch.median(torch.FloatTensor(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.FloatTensor(running_median))
                        running_median = []

        # labels come from p
        pred_label_e = self.label_p[e_idx] if self.label_p is not None else None

        return e_idx, pred_label_e

    @staticmethod
    def update_p_base(e_idx, data_p, data_e):
        """ base function to update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            data_p (pytorch Tensor): cluster centroids, p
            data_e (pytorch Tensor): empirical samples, e
            p0 (pytorch Tensor): iteration index

        Returns:
            p0 (pytorch Tensor): new p
            max_change_pct (float): max_change
        """

        p0 = torch.zeros(data_p.shape).double()
        num_p = data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(e_idx, minlength=num_p).double()
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(e_idx, weights=data_e[:, i], minlength=num_p).double() / (bincount+eps)
            change_pct = torch.max(torch.abs((data_p[:, i] - p_target) / (data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target

        # replace nan by original data TODO replace nan by nn barycenter?
        mask = torch.isnan(p0).any(dim=1)
        p0[mask] = data_p[mask].clone()

        return p0, max_change_pct

    def update_p(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            iter_p (int): iteration index

        Returns:
            (bool): convergence or not, determined by max p change
        """

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)
        self.data_p = p0

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False


class VotReg(Vot):
    """ variational optimal transportation with regularization on sample supports"""

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 weight_p=None, weight_e=None, thres=1e-3, verbose=True, device='cpu'):
        super(VotReg, self).__init__(data_p, data_e, label_p=label_p, label_e=label_e,
                                     weight_p=weight_p, weight_e=weight_e, thres=thres, verbose=verbose, device=device)

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
        self.data_p.requires_grad_(True)
        e_idx, pred_label_e = None, None
        for iter_p in range(max_iter_p):
            dist = torch.cdist(self.data_p, self.data_e) ** 2
            e_idx, pred_label_e = self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay, early_stop=early_stop)
            # reg = reg / 20 * (20 - iter_p)
            # reg /= 1
            if self.update_p(e_idx, iter_p, reg_type, reg):
                break
        return e_idx, pred_label_e

    def update_p(self, e_idx, iter_p=0, reg_type=0, reg=0.01):
        """ update p

        Args:
            e_idx (torch Tensor): assignment of e to p
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
        elif reg_type == 3 or reg_type == 'triplet':
            return self.update_p_reg_triplet(e_idx, iter_p, reg)
        else:
            return self.update_p_noreg(e_idx, iter_p)

    def update_p_noreg(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (torch Tensor): assignment of e to p
            iter_p (int): iteration index

        Returns:
            bool: convergence or not, determined by max p change
        """

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p, self.data_e)
        self.data_p = p0

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False

    def update_p_reg_potential(self, e_idx, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            e_idx (torch Tensor): assignment of e to p
            iter_p (int): index of the iteration of updating p
            reg (float): regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, label=None, reg=0.1):
            """ objective function incorporating labels

            Args:
                p  pytorch floattensor:   p
                p0 pytorch floattensor:  centroids of e
                label pytorch inttensor: labels of p
                reg float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + reg * sum(1(li == lj)*|pi-pj|^2)
            """

            reg_term = 0.0
            for l in torch.unique(label):
                p_sub = p[label == l, :]
                reg_term += torch.pow(torch.pdist(p_sub, p=2), 2).sum()

            return torch.sum((p - p0) ** 2.0) + reg * reg_term

        if torch.unique(self.label_p).size == 1:
            warnings.warn("All known samples belong to the same class")

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p.detach(), self.data_e)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # regularize
        optimizer = optim.SGD([self.data_p], lr=0.05)
        for _ in range(10):
            optimizer.zero_grad()
            loss = f(self.data_p, p0, self.label_p, reg=reg)
            loss.backward()
            optimizer.step()
        # return convergence or not
        return True if max_change_pct < self.thres else False

    def update_p_reg_transform(self, e_idx, iter_p=0, reg=0.01):
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

        assert self.data_p.shape[1] == 3 or self.data_p.shape[1] == 2, "dim has to be 2 or 3 for geometric transformation"

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p.detach(), self.data_e)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # pt = utils.estimate_transform_target_pytorch(self.y.detach(), p0)
        pt = utils.estimate_transform_target(self.data_p.detach(), p0)
        pt = torch.tensor(pt, device=self.device)
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



        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt
        # return convergence
        return True if max_change_pct < self.thres else False

    def update_p_reg_triplet(self, e_idx, iter_p=0, reg=0.01, margin=0.1):
        """ update each p to the centroids of its cluster,
            regularized by triplet loss

        Args:
            e_idx (torch Tensor): assignment of e to p
            iter_p (int): index of the iteration of updating p
            reg (float): regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """

        def f(p, p0, mask, reg=0.1, margin=0.1):
            """ objective function incorporating labels

            Args:
                p  pytorch floattensor:   p
                p0 pytorch floattensor:  centroids of e
                label pytorch inttensor: labels of p
                reg float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + reg * sum(1(li == lj)*|pi-pj|^2)
            """

            dists = (p[None, :] - p[:, None]).pow(2).sum(2)

            positive = dists * mask
            negative = dists * (1 - mask)

            reg_term = torch.nn.functional.relu(positive - negative + margin).sum()
            data_term = ((p - p0) ** 2.0).sum()

            return data_term + reg * reg_term

        if torch.unique(self.label_p).size == 1:
            warnings.warn("All known samples belong to the same class")

        p0, max_change_pct = self.update_p_base(e_idx, self.data_p.detach(), self.data_e)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}".format(iter_p, max_change_pct))

        # regularize
        optimizer = optim.SGD([self.data_p], lr=0.05)
        mask = (self.label_p[None, :] == self.label_p[:, None]).double()
        for _ in range(10):
            optimizer.zero_grad()
            loss = f(self.data_p, p0, mask, reg=reg, margin=margin)
            loss.backward()
            optimizer.step()
        # return convergence or not
        return True if max_change_pct < self.thres else False


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

        self.dist = torch.cdist(self.y, self.x.double()).to(device)

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


class VWB:
    def __init__(self, data_p, data_e: [], label_p=None, label_e=None,
                 weight_p=None, weight_e=None, weight_k=None, thres=1e-4, verbose=True, device='cpu'):

        # deep copy all the data?
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.clone()

        num_p = data_p.shape[0]
        n = len(data_e)  # number of empirical distributions

        self.label_p = label_p
        self.label_e = label_e

        self.thres = thres
        self.verbose = verbose
        self.device = device

        self.weight_k = weight_k if weight_k is not None else torch.ones(n).double().to(device) / n
        self.weight_p = weight_p if weight_p is not None else torch.ones(num_p).double().to(device) / num_p
        if weight_e is not None:
            self.weight_e = weight_e
        else:
            self.weight_e = []
            for i in range(n):
                data = data_e[i]
                num_e = data.shape[0]
                self.weight_e.append(torch.ones(num_e).double().to(device) / num_e)

        utils.assert_boundary(self.data_p)
        for i in range(n):
            utils.assert_boundary(self.data_e[i])

    def cluster(self, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0, reg=0.):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay
            reg      (float): for regularized k-means

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        e_idx_return, pred_label_e_return = [], []
        n = len(self.data_e)
        dhss = []
        e_idxss = []
        for iter_p in range(max_iter_p):
            e_idx, pred_label_e = [], []
            for i in range(n):
                # if self.verbose:
                print("solving marginal #" + str(i))
                dist = (torch.cdist(self.data_p, self.data_e[i]) ** 2).double().to(self.device)
                idx, pred_label, dhs, e_idxs = self.update_map(i, dist, max_iter_h, lr=lr, lr_decay=lr_decay, beta=beta, early_stop=early_stop, reg=reg)
                dhss.append(dhs)
                e_idxss.append(e_idxs)
                e_idx.append(idx)
                pred_label_e.append(pred_label)
            if self.update_p(e_idx, iter_p):
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
                break
            if iter_p == max_iter_p - 1:
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
        output = dict()
        output['idx'] = e_idx_return
        output['pred_label_e'] = pred_label_e_return
        output['dhss'] = dhss
        output['idxs'] = e_idxss

        # compute WD
        wd = 0
        for e_idx, data_e, weight_e in zip(e_idx_return, self.data_e, self.weight_e):
            tmp = self.data_p[e_idx, :]
            tmp -= data_e
            tmp = tmp ** 2
            wd += torch.sum(torch.sum(tmp, dim=1) * weight_e)

        output['wd'] = 2 * wd
        return output

    def update_map(self, n, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, early_stop=200, reg=0.):
        """ update assignment of each e as the ot_map to p

        Args:
            dist (pytorch Tensor): dist matrix across p and e
            max_iter   (int): max num of iterations
            lr       (float): gradient descent learning rate
            beta     (float): GD momentum
            lr_decay (int): learning rate decay frequency
            early_stop (int): early_stop check frequency
            reg (float): for regularized k-means

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0

        dhs = []
        e_idxs = []

        dist_original = 0 if reg == 0 else dist.clone()

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = torch.argmin(dist, dim=0)
            mass_p = torch.bincount(e_idx, weights=self.weight_e[n], minlength=num_p).double().to(self.device)
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            dhs.append((mass_p - self.weight_p).clone())
            e_idxs.append(e_idx.clone())
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dh *= lr
            dist += dh[:, None]

            # check if converge
            if self.verbose and i % 1000 == 0:
                print(dh)
            max_change = torch.max((mass_p - self.weight_p)/self.weight_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 1 == 0) or i % 100 == 0):
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
                            torch.abs(torch.median(torch.FloatTensor(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.FloatTensor(running_median))
                        running_median = []

        # labels come from p
        if reg != 0.:
            e_idx = torch.argmin(reg / (1 + reg) * dist + 1 / (1 + reg) * dist_original, dim=0)
        pred_label_e = self.label_p[n][e_idx] if self.label_p is not None else None

        return e_idx, pred_label_e, dhs, e_idxs

    @staticmethod
    def update_p_base(e_idx, data_p, data_e):
        """ base function to update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            data_p (pytorch Tensor): cluster centroids, p
            data_e (pytorch Tensor): empirical samples, e
            p0 (pytorch Tensor): iteration index

        Returns:
            p0 (pytorch Tensor): new p
            max_change_pct (float): max_change
        """

        p0 = torch.zeros(data_p.shape).double().to(data_p.device)
        num_p = data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(e_idx, minlength=num_p).double().to(data_p.device)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try a different lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(e_idx, weights=data_e[:, i], minlength=num_p).double().to(data_p.device) / (bincount+eps)
            change_pct = torch.max(torch.abs((data_p[:, i] - p_target) / (data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target

        # replace nan by original data TODO replace nan by nn barycenter?
        mask = torch.isnan(p0).any(dim=1)
        p0[mask] = data_p[mask].clone()

        return p0, max_change_pct

    def update_p(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            iter_p (int): iteration index

        Returns:
            (bool): convergence or not, determined by max p change
        """

        n = len(self.data_e)
        num_p, d = self.data_p.shape[0], self.data_p.shape[1]
        max_change_pct = 1e9

        p = torch.zeros((n, num_p, d), dtype=torch.double, device=self.device)
        for i in range(n):
            p[i], change = self.update_p_base(e_idx[i], self.data_p, self.data_e[i])
            max_change_pct = max(max_change_pct, change)

        self.data_p = torch.sum(p * self.weight_k[:, None, None], dim=0)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False


class RegVWB(VWB):
    def cluster(self, reg_type=0, reg=0.01, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        e_idx_return, pred_label_e_return = [], []
        n = len(self.data_e)
        dhss = []
        e_idxss = []
        for iter_p in range(max_iter_p):
            e_idx, pred_label_e = [], []
            for i in range(n):
                # if self.verbose:
                print("solving marginal #" + str(i))
                dist = (torch.cdist(self.data_p, self.data_e[i]) ** 2).double().to(self.device)
                idx, pred_label, dhs, e_idxs = self.update_map(i, dist, max_iter_h, lr=lr, lr_decay=lr_decay, beta=beta, early_stop=early_stop)
                dhss.append(dhs)
                e_idxss.append(e_idxs)
                e_idx.append(idx)
                pred_label_e.append(pred_label)
            if self.update_p(e_idx, iter_p, reg=reg):
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
                break
            if iter_p == max_iter_p - 1:
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
        output = dict()
        output['idx'] = e_idx_return
        output['pred_label_e'] = pred_label_e_return
        output['dhss'] = dhss
        output['idxs'] = e_idxss

        # compute WD
        wd = 0
        for e_idx, data_e, weight_e in zip(e_idx_return, self.data_e, self.weight_e):
            tmp = self.data_p[e_idx, :]
            tmp -= data_e
            tmp = tmp ** 2
            wd += torch.sum(torch.sum(tmp, dim=1) * weight_e)

        output['wd'] = 2 * wd
        return output

    def update_p(self, e_idx, iter_p=0, reg=0.01):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            iter_p (int): iteration index

        Returns:
            (bool): convergence or not, determined by max p change
        """

        n = len(self.data_e)
        num_p, d = self.data_p.shape[0], self.data_p.shape[1]
        max_change_pct = 1e9

        p0 = torch.zeros((n, num_p, d), dtype=torch.double, device=self.device)
        pt = torch.zeros((n, num_p, d), dtype=torch.double, device=self.device)
        for i in range(n):
            p0[i], change = self.update_p_base(e_idx[i], self.data_p, self.data_e[i])
            max_change_pct = max(max_change_pct, change)
            tmp = utils.estimate_transform_target(self.data_p.detach(), p0[i].detach())
            pt[i] = torch.tensor(tmp, device=self.device)

        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt

        self.data_p = torch.sum(pt * self.weight_k[:, None, None], dim=0)

        if self.verbose:
            print("it {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False


class SVWB (VWB):
    def __init__(self, data_p, data_e: [], label_p=None, label_e=None,
                 weight_p=None, weight_e=None, weight_k=None, thres=1e-4, verbose=True, device='cpu'):

        # deep copy all the data?
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.clone()

        num_p = data_p.shape[0]
        n = len(data_e)  # number of empirical distributions

        self.label_p = label_p
        self.label_e = label_e

        self.thres = thres
        self.verbose = verbose
        self.device = device

        self.weight_k = weight_k if weight_k is not None else torch.ones(n).double() / n
        self.weight_p = weight_p if weight_p is not None else torch.ones(num_p).double() / num_p
        if weight_e is not None:
            self.weight_e = weight_e
        else:
            self.weight_e = []
            for i in range(n):
                data = data_e[i]
                num_e = data.shape[0]
                self.weight_e.append(torch.ones(num_e).double() / num_e)

        utils.assert_boundary(self.data_p)
        for i in range(n):
            utils.assert_boundary(self.data_e[i])

    def cluster(self, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        e_idx_return, pred_label_e_return = [], []
        n = len(self.data_e)
        dhss = []
        e_idxss = []

        for iter_p in range(max_iter_p):
            e_idx, pred_label_e = [], []
            for i in range(n):
                # if self.verbose:
                print("solving marginal #" + str(i))
                dist = torch.mm(self.data_p, self.data_e[i].T)
                # dist = torch.cdist(self.y, self.x[i]) ** 2
                idx, pred_label, dhs, e_idxs = self.update_map(i, dist, max_iter_h, lr=lr, lr_decay=lr_decay, beta=beta, early_stop=early_stop)
                dhss.append(dhs)
                e_idxss.append(e_idxs)
                e_idx.append(idx)
                pred_label_e.append(pred_label)
            if self.update_p(e_idx, iter_p):
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
                break
            if iter_p == max_iter_p - 1:
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
        output = dict()
        output['idx'] = e_idx_return
        output['pred_label_e'] = pred_label_e_return
        output['dhss'] = dhss
        output['idxs'] = e_idxss
        return output

    def update_map(self, n, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, early_stop=200):
        """ update assignment of each e as the ot_map to p

        Args:
            dist (pytorch Tensor): dist matrix across p and e
            max_iter   (int): max num of iterations
            lr       (float): gradient descent learning rate
            beta     (float): GD momentum
            lr_decay (int): learning rate decay frequency
            early_stop (int): early_stop check frequency

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0
        h = torch.ones(num_p).double().to(self.device)
        dhs = []
        e_idxs = []

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = torch.argmin(dist / torch.cos(h)[:, None], dim=0)
            mass_p = torch.bincount(e_idx, weights=self.weight_e[n], minlength=num_p).double()
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            dhs.append((mass_p - self.weight_p).clone())
            e_idxs.append(e_idx.clone())
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dh *= lr
            h += dh

            # check if converge
            if self.verbose and i % 1000 == 0:
                print(dh)
            max_change = torch.max((mass_p - self.weight_p)/self.weight_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 1 == 0) or i % 100 == 0):
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
                            torch.abs(torch.median(torch.FloatTensor(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.FloatTensor(running_median))
                        running_median = []

        # labels come from p
        pred_label_e = self.label_p[n][e_idx] if self.label_p is not None else None

        return e_idx, pred_label_e, dhs, e_idxs


class UVWB(VWB):
    def __init__(self, data_p, data_e: [], label_p=None, label_e=None,
                 weight_p=None, weight_e=None, weight_k=None, thres=1e-4, verbose=True, device='cpu'):
        # deep copy all the data?
        self.data_p = data_p
        self.data_e = data_e
        self.data_p_original = self.data_p.clone()

        num_p = data_p.shape[0]
        n = len(data_e)  # number of empirical distributions

        self.label_p = label_p
        self.label_e = label_e

        self.verbose = verbose
        self.device = device

        self.weight_k = weight_k if weight_k is not None else torch.ones(n).double() / n
        self.weight_p = weight_p if weight_p is not None else torch.ones(num_p).double() / num_p
        self.weight_p_sum = torch.sum(self.weight_p)
        if weight_e is not None:
            self.weight_e = weight_e
        else:
            self.weight_e = []
            for i in range(n):
                data = data_e[i]
                num_e = data.shape[0]
                self.weight_e.append(torch.ones(num_e).double())
        self.weight_e_sum = [torch.sum(w) for w in self.weight_e]
        self.thres = thres * abs(self.weight_p_sum - max(self.weight_e_sum))
        utils.assert_boundary(self.data_p)
        for i in range(n):
            utils.assert_boundary(self.data_e[i])

    def cluster(self, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_p : update p
        update_map: compute optimal transportation
        """

        dhss = []
        e_idxss = []

        lrs = [lr / torch.abs(weight - self.weight_p_sum) for weight in self.weight_e_sum]

        e_idx_return, pred_label_e_return = [], []
        n = len(self.data_e)
        for iter_p in range(max_iter_p):
            e_idx, pred_label_e = [], []
            for i in range(n):
                lr = lrs[i]
                # if self.verbose:
                print("solving marginal #" + str(i))
                dist = torch.cdist(self.data_p, self.data_e[i]) ** 2
                idx, pred_label, dhs, e_idxs = self.update_map(i, dist, max_iter_h, lr=lr, lr_decay=lr_decay, beta=beta, early_stop=early_stop)
                dhss.append(dhs)
                e_idxss.append(e_idxs)
                # pred_label_es.append(pred_label)
                e_idx.append(idx)
                pred_label_e.append(pred_label)
            if self.update_p(e_idx, iter_p):
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
                break
            if iter_p == max_iter_p - 1:
                e_idx_return, pred_label_e_return = e_idx, pred_label_e
        output = dict()
        output['idx'] = e_idx_return
        output['pred_label_e'] = pred_label_e_return
        output['dhss'] = dhss
        output['idxs'] = e_idxss
        return output

    def update_map(self, n, dist, max_iter=3000, lr=0.5, beta=0, lr_decay=200, early_stop=200):
        """ update assignment of each e as the ot_map to p

        Args:
            dist (pytorch Tensor): dist matrix across p and e
            max_iter   (int): max num of iterations
            lr       (float): gradient descent learning rate
            beta     (float): GD momentum
            lr_decay (int): learning rate decay frequency
            early_stop (int): early_stop check frequency

        Returns:
            idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        dh = 0
        e_idx = None
        running_median, previous_median = [], 0
        dhs = []
        e_idxs = []

        for i in range(max_iter):
            # find nearest p for each e and add mass to p
            e_idx = torch.argmin(dist, dim=0)
            mass_p = torch.bincount(e_idx, weights=self.weight_e[n], minlength=num_p).double()
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            dhs.append((mass_p - self.weight_p).clone())
            e_idxs.append(e_idx.clone())
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dh *= lr
            dist += dh[:, None]

            # check if converge
            if self.verbose and i % 1000 == 0:
                print(dh)
            max_change = torch.max((mass_p - self.weight_p)/self.weight_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and ((i < 100 and i % 1 == 0) or i % 100 == 0):
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
                            torch.abs(torch.median(torch.FloatTensor(running_median))-previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.FloatTensor(running_median))
                        running_median = []

        # labels come from p
        pred_label_e = self.label_p[n][e_idx] if self.label_p is not None else None

        return e_idx, pred_label_e, dhs, e_idxs


class ICPVWB(VWB):
    def cluster(self, reg_type=0, reg=0.01, lr=0.5, max_iter_p=10, max_iter_h=3000, lr_decay=200, early_stop=-1, beta=0):
        """ compute Wasserstein clustering

        Args:
            reg_type   (int): specify regulazation term, 0 means no regularization
            reg        (int): regularization weight
            max_iter_p (int): max num of iteration of clustering
            max_iter_h (int): max num of updating h
            lr       (float): GD learning rate
            lr_decay (float): learning rate decay

        Returns:
            e_idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p

        See Also
        --------
        update_y : update p
        update_map: compute optimal transportation
        """

        e_idx_return, pred_label_e_return = [], []
        n = len(self.data_e)
        # dhss = []
        # e_idxss = []
        for iter_p in range(max_iter_p):
            e_idx, pred_label_e = [], []
            for i in range(n):
                # if self.verbose:
                print("solving marginal #" + str(i))
                dist = (torch.cdist(self.data_p, self.data_e[i]) ** 2).to(self.device)
                output = self.update_map(i, dist, max_iter_h, lr=lr, lr_decay=lr_decay, beta=beta, early_stop=early_stop)
                # dhss.append(dhs)
                # e_idxss.append(e_idxs)
                # e_idx.append(idx)
                # pred_label_e.append(pred_label)
                e_idx.append(output[0])
            if self.update_e(e_idx, iter_p):
                # e_idx_return, pred_label_e_return = e_idx, pred_label_e
                break
            # if iter_p == max_iter_p - 1:
                # e_idx_return, pred_label_e_return = e_idx, pred_label_e
        output = dict()
        # output['e_idx'] = e_idx_return
        # output['pred_label_e'] = pred_label_e_return
        # output['dhss'] = dhss
        # output['e_idxss'] = e_idxss

        # compute WD
        wd = 0
        for e_idx, data_e, weight_e in zip(e_idx_return, self.data_e, self.weight_e):
            tmp = self.data_p[e_idx, :]
            tmp -= data_e
            tmp = tmp ** 2
            wd += torch.sum(torch.sum(tmp, dim=1) * weight_e)

        output['wd'] = 2 * wd
        return output

    def update_e(self, e_idx, iter_p=0):
        """ update each p to the centroids of its cluster

        Args:
            e_idx (pytorch Tensor): assignment of e to p
            iter_p (int): iteration index

        Returns:
            (bool): convergence or not, determined by max p change
        """

        n = len(self.data_e)
        num_p, d = self.data_p.shape[0], self.data_p.shape[1]
        max_change_pct = 1e9

        p = torch.zeros((n, num_p, d), device=self.device)
        for i in range(n):
            p[i], change = self.update_p_base(e_idx[i], self.data_p, self.data_e[i])
            max_change_pct = max(max_change_pct, change)

        self.data_p = torch.sum(p * self.weight_k[:, None, None], dim=0)

        for i in range(n):
            r, t = utils.estimate_inverse_transform(self.data_p.detach(), p[i].detach())
            r = torch.tensor(r, device=self.device)
            t = torch.tensor(t, device=self.device)
            self.data_e[i] = (torch.mm(r, self.data_e[i].t()) + t).t()
            print(r.inverse())

        if self.verbose:
            print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < self.thres else False

