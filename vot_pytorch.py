# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 30th 2019


import imageio
import utils
import torch
import warnings
import torch.optim as optim
import numpy as np

class Vot:
    """ variational optimal transportation """

    def __init__(self, data_p, data_e, label_p=None, label_e=None, mass_p=None, mass_e=None, thres=1e-5, ratio=100, verbose=True, device='cpu'):
        """ set up parameters
        Args:
            thres float: threshold to break loops
            ratio float: the ratio of num of e to the num of p
            data_p pytorch floattensor: initial coordinates of p
            label_p pytorch inttensor: labels of p
            mass_p pytorch floattensor: weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            ratio    float: ratio of num_e to num_p
            verbose   bool: console output verbose flag
            num_p      int: number of p
            X_p    pytorch floattensor: coordinates of p
            y_p    pytorch inttensor: labels of p
            mass_p pytorch floattensor: mass of clusters of p

        """
        if not isinstance(data_p, torch.Tensor):
            raise Exception('data_p is not a pytorch tensor')
        if not isinstance(data_e, torch.Tensor):
            raise Exception('data_e is not a pytorch tensor')
        self.data_p = data_p.float().to(device)
        self.data_e = data_e.float().to(device)
        self.data_p_original = self.data_p.clone()
        self.data_e_original = self.data_e.clone()

        self.data_p.requires_grad_(True)

        num_p = data_p.shape[0]
        num_e = data_e.shape[0]

        if label_p is not None and not isinstance(label_p, torch.Tensor):
            raise Exception('label_p is not a pytorch tensor')
        if label_e is not None and not isinstance(label_e, torch.Tensor):
            raise Exception('label_e is not a pytorch tensor')
        self.label_p = label_p.int()
        self.label_e = label_e.int()

        if mass_p is not None and not isinstance(mass_p, torch.Tensor):
            raise Exception('label_p is not a pytorch tensor')
        if mass_p is not None:
            self.p_dirac = mass_p
        else:
            self.p_dirac = torch.ones(num_p).float().to(device) / num_p

        self.thres = thres
        self.verbose = verbose
        self.ratio = ratio
        self.device = device

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.mass_p = torch.zeros(num_p).float().to(self.device)

        self.p_dirac = mass_p if mass_p is not None else torch.ones(num_p).float().to(self.device) / num_p
        self.mass_e = mass_e if mass_e is not None else torch.ones(num_e).float().to(self.device) / num_e

        self.h = torch.zeros(num_p).float().to(self.device)

        assert torch.max(self.data_p) <= 1 and torch.min(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

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
            base_dist = torch.cdist(self.data_p, self.data_e, p=2).float().to(self.device) ** 2
            self.update_map(base_dist, max_iter_h, lr=lr)
            if self.update_p(iter_p, reg_type, reg):
                break

    def update_map(self, base_dist, max_iter, lr=0.2, beta=0.9, lr_decay=50):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            bool: convergence or not, determined by max derivative change
        """
        num_p = self.data_p.shape[0]

        self.h[self.h != 0] = 0

        for i in range(max_iter):
            # update dist matrix
            dist = base_dist - self.h[:, None]
            # find nearest p for each e and add mass to p
            self.e_idx = torch.argmin(dist, dim=0)
            # labels come from centroids
            self.e_predict = self.label_p[self.e_idx]
            self.mass_p = torch.bincount(self.e_idx, weights=self.mass_e, minlength=num_p)
            # update gradient and h
            dh = self.mass_p - self.p_dirac

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.p_dirac)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            self.h -= lr * dh

            # check if converge
            max_change = torch.max(dh / self.mass_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:.2f}%".format(i, max_change))

            if max_change <= 1:
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
        num_p = self.data_p.shape[0]

        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(self.e_idx, minlength=num_p).float().to(self.device)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p).float().to(self.device) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / self.data_p[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            self.data_p[:, i] = p_target
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

            reg_term = 0.0
            for l in torch.unique(label):
                p_sub = p[label == l, :]
                reg_term += torch.pow(torch.pdist(p_sub, p=2), 2).sum()

            return torch.sum((p - p0) ** 2.0) + reg * reg_term

        if torch.unique(self.label_p).size == 1:
            warnings.warn("All known samples belong to the same class")

        p0 = torch.zeros_like(self.data_p)
        num_p = self.data_p.shape[0]
        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(self.e_idx).float()
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Abort this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        for i in range(p0.shape[1]):
            # update p to the centroid of their correspondences
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / self.data_p[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        # regularize
        optimizer = optim.Adam([self.data_p], lr=0.05)
        for _ in range(10):
            optimizer.zero_grad()
            loss = f(self.data_p, p0, self.label_p, reg=0.1)
            loss.backward()
            optimizer.step()

        # return max change
        return True if max_change_pct < 0.01 else False

    def update_p_reg_transform(self, iter_p, reg=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

        Returns:
            bool: convergence or not, determined by max p change
        """
        # TODO transformation for each class?
        def f(p, p0, pt, label=None, reg=0.01):
            """ objective function incorporating labels

            Args:
                p  np.array(np,dim):   p
                p0 np.array(np,dim):  centroids of e
                label np.array(np,): labels of p
                reg float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + reg * sum(1(li == lj)*|pi-pj|^2)
            """

            return torch.mean((p - p0) ** 2) + reg * torch.mean((p - pt) ** 2)
            # return torch.mean((p - p0) ** 2)

        p0 = torch.zeros_like(self.data_p)
        num_p = self.data_p.shape[0]
        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(self.e_idx).float()
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Aborting this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        # update p to the centroid of their correspondences
        for i in range(p0.shape[1]):
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / self.data_p[:, i]))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        # pt = self.data_p.clone().detach().cpu().numpy()
        # pt = utils.estimate_transform_target_pytorch(pt, p0.cpu().numpy())
        # pt = torch.from_numpy(pt).float().to(self.device)

        pt = self.data_p.clone().detach()
        pt = utils.estimate_transform_target_pytorch(pt, p0)

        # regularize
        optimizer = optim.Adam([self.data_p], lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            loss = f(self.data_p, p0, pt, self.label_p, reg=reg)
            loss.backward()
            optimizer.step()

        # return max change
        return True if max_change_pct < 0.01 else False


class VotAP:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the area samples

    def __init__(self, data, label=None, mass_p=None, thres=1e-5, ratio=100, verbose=True, device='cpu'):
        """ set up parameters
        Args:
            thres float: threshold to break loops
            ratio float: the ratio of num of e to the num of p
            data pytorch floattensor: initial coordinates of p
            label pytorch inttensor: labels of p
            mass_p pytorch floattensor: weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            ratio    float: ratio of num_e to num_p
            verbose   bool: console output verbose flag
            num_p      int: number of p
            X_p    pytorch floattensor: coordinates of p
            y_p    pytorch inttensor: labels of p
            mass_p pytorch floattensor: mass of clusters of p

        """
        if not isinstance(data, torch.Tensor):
            raise Exception('input is not a pytorch tensor')
        self.data_p = data
        self.data_p_original = self.data_p.clone()
        num_p = data.shape[0]

        if label and not isinstance(label, torch.Tensor):
            raise Exception('label is neither a numpy array not a pytorch tensor')
        self.label_p = label

        if mass_p and not isinstance(mass_p, torch.Tensor):
            raise Exception('label is neither a numpy array not a pytorch tensor')
        if mass_p:
            self.p_dirac = mass_p
        else:
            self.p_dirac = torch.ones(num_p).float().to(device) / num_p

        self.thres = thres
        self.verbose = verbose
        self.ratio = ratio
        self.device = device

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "p_dirac"
        self.mass_p = torch.zeros(num_p).float().to(self.device)

        assert torch.max(self.data_p) <= 1 and torch.min(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

    def map(self, sampling='unisquare', plot_filename=None, beta=0.9, max_iter=1000, lr=0.2, lr_decay=100):
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
            data_e pytorch floattensor: coordinates of e
            label_e pytorch inttensor: label of e
            base_dist pytorch floattensor: pairwise distance between p and e
            h  pytorch floattensor: VOT optimizer, "height vector
            dh  pytorch floattensor: gradient of h
            max_change pytorch floattensor: maximum gradient change
            max_change_pct pytorch floattensor: relative maximum gradient change
            imgs list: list of plots to show mapping progress
            e_idx pytorch inttensor: p index of every e

        :return:
        """
        num_p = self.data_p.shape[0]
        num_e = self.ratio * num_p
        dim = self.data_p.shape[1]
        self.data_e, _ = utils.random_sample(num_e, dim, sampling=sampling)
        self.data_e = torch.from_numpy(self.data_e).float().to(self.device)

        base_dist = torch.cdist(self.data_p, self.data_e, p=2).float().to(self.device)**2
        self.e_idx = torch.argmin(base_dist, dim=0)
        h = torch.zeros(num_p).float().to(self.device)
        imgs = []
        dh = torch.zeros(num_p).float().to(self.device)

        for i in range(max_iter):
            dist = base_dist - h[:, None]

            # find nearest p for each e
            self.e_idx = torch.argmin(dist, dim=0)

            # calculate total mass of each cell
            self.mass_p = torch.bincount(self.e_idx, minlength=num_p).float().to(self.device) / num_e

            # labels come from centroids
            if self.label_p:
                self.label_e = self.label_p[self.e_idx]

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.p_dirac)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            h -= lr * dh

            # check if converge
            max_change = torch.max(dh / self.mass_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:.2f}%".format(i, max_change))
            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename:
                fig = utils.plot_map(self.data_e.cpu().numpy(), self.e_idx.cpu().numpy() / (num_p - 1))
                img = utils.fig2data(fig)
                imgs.append(img)
            if max_change <= 1:
                break
        if plot_filename and imgs:
            imageio.mimsave(plot_filename, imgs, fps=4)

        # update coordinates of p
        bincount = torch.bincount(self.e_idx).float()
        if 0 in bincount:
            print('Empty cluster found, optimal transport did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p).float() / bincount
