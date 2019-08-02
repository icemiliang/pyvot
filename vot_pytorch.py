# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 30th 2019


import imageio
import utils
import torch
import warnings
import torch.optim as optim


class Vot:
    """ variational optimal transportation """

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 mass_p=None, mass_e=None, thres=1e-5, verbose=True, device='cpu'):
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
            data_p     pytorch floattensor: coordinates of p
            data_e     pytorch floattensor: coordinates of e
            label_p    pytorch inttensor: labels of p
            label_e    pytorch inttensor: labels of e
            mass_p     pytorch floattensor: mass of clusters of p
            mass_e     pytorch floattensor: mass of e
            dirac_p    pytorch floattensor: dirac measure of p
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
        self.label_p = label_p
        self.label_e = label_e

        if mass_p is not None and not isinstance(mass_p, torch.Tensor):
            raise Exception('label_p is not a pytorch tensor')
        if mass_p is not None:
            self.dirac_p = mass_p
        else:
            self.dirac_p = torch.ones(num_p).float().to(device) / num_p

        self.thres = thres
        self.verbose = verbose
        self.device = device

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "dirac_p"
        self.mass_p = torch.zeros(num_p).float().to(self.device)

        self.dirac_p = mass_p if mass_p is not None else torch.ones(num_p).float().to(self.device) / num_p
        self.mass_e = mass_e if mass_e is not None else torch.ones(num_e).float().to(self.device) / num_e

        assert torch.max(self.data_p) <= 1 and torch.min(self.data_p) >= -1,\
            "Input output boundary (-1, 1)."

    def cluster(self, lr=0.2, max_iter_p=10, max_iter_h=2000, lr_decay=200):
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
            dist = torch.cdist(self.data_p, self.data_e, p=2).float().to(self.device) ** 2
            self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay)
            if self.update_p(iter_p):
                break

    def update_map(self, dist, max_iter, lr=0.2, beta=0.9, lr_decay=200):
        """ update each p to the centroids of its cluster

        Args:
            dist    pytorch floattensor: dist matrix across p and e
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
            self.e_idx = torch.argmin(dist, dim=0)
            self.mass_p = torch.bincount(self.e_idx, weights=self.mass_e, minlength=num_p)

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.dirac_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dist += lr * dh[:, None]

            # check if converge
            max_change = torch.max(dh / self.dirac_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100

            if self.verbose and i % 10 == 0:
                print("{0:d}: max gradient {1:.2f}%".format(i, max_change))

            if max_change <= 1:
                break

        running_median, previous_median = [], 0

        for i in range(max_iter):
            e_idx = torch.argmin(dist, dim=0)
            mass_p = torch.bincount(self.e_idx, weights=self.mass_e, minlength=num_p)
            dh = beta * dh + (1 - beta) * (mass_p - self.dirac_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            dist += lr * dh[:, None]
            max_change = torch.max(dh / self.dirac_p)
            if max_change.numel() > 1:
                max_change = max_change[0]
            max_change *= 100
            if self.verbose and i % 10 == 0:
                print("{0:d}: mass diff {1:.2f}%".format(i, max_change))

            if max_change < 1:
                if self.verbose:
                    print("{0:d}: mass diff {1:.2f}%".format(i, max_change))
                break

            # early stop if loss does not decrease TODO better way to early stop?
            running_median.append(max_change)
            if len(running_median) >= 100:
                if previous_median != 0 and \
                        torch.abs(
                            torch.median(torch.FloatTensor(running_median)) - previous_median) / previous_median < 0.01:
                    print("loss saturated, early stopped")
                    break
                else:
                    previous_median = torch.median(torch.FloatTensor(running_median))
                    running_median = []
        # labels come from centroids
        if self.label_p is not None:
            self.e_predict = self.label_p[self.e_idx]

    def update_p(self, iter_p=0):

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
            # return False
        eps = 1e-8
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p).float().to(self.device) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            self.data_p[:, i] = p_target

        # replace nan by original data
        mask = torch.isnan(self.data_p).any(dim=1)
        self.data_p[mask] = self.data_p_original[mask].clone()
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < 0.01 else False


class VotReg:
    """ variational optimal transportation """

    def __init__(self, data_p, data_e, label_p=None, label_e=None,
                 mass_p=None, mass_e=None, thres=1e-5, verbose=True, device='cpu'):
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
            data_p     pytorch floattensor: coordinates of p
            data_e     pytorch floattensor: coordinates of e
            label_p    pytorch inttensor: labels of p
            label_e    pytorch inttensor: labels of e
            mass_p     pytorch floattensor: mass of clusters of p
            mass_e     pytorch floattensor: mass of e
            dirac_p    pytorch floattensor: dirac measure of p
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
        self.label_p = label_p
        self.label_e = label_e

        if mass_p is not None and not isinstance(mass_p, torch.Tensor):
            raise Exception('label_p is not a pytorch tensor')
        if mass_p is not None:
            self.dirac_p = mass_p
        else:
            self.dirac_p = torch.ones(num_p).float().to(device) / num_p

        self.thres = thres
        self.verbose = verbose
        self.device = device

        # "mass_p" is the sum of its corresponding e's weights, its own weight is "dirac_p"
        self.mass_p = torch.zeros(num_p).float().to(self.device)

        self.dirac_p = mass_p if mass_p is not None else torch.ones(num_p).float().to(self.device) / num_p
        self.mass_e = mass_e if mass_e is not None else torch.ones(num_e).float().to(self.device) / num_e

        assert torch.max(self.data_p) <= 1 and torch.min(self.data_p) >= -1,\
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
            dist = torch.cdist(self.data_p, self.data_e, p=2).float().to(self.device) ** 2
            self.update_map(dist, max_iter_h, lr=lr, lr_decay=lr_decay)
            if self.update_p(iter_p, reg_type, reg):
                break

    def update_map(self, dist, max_iter, lr=0.2, beta=0.9, lr_decay=200):
        """ update each p to the centroids of its cluster

        Args:
            dist    pytorch floattensor: dist matrix across p and e
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
            self.e_idx = torch.argmin(dist, dim=0)
            self.mass_p = torch.bincount(self.e_idx, weights=self.mass_e, minlength=num_p)

            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (self.mass_p - self.dirac_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.5
            # update dist matrix
            dist += lr * dh[:, None]

            # check if converge
            max_change = torch.max(dh / self.dirac_p)
            if max_change.numel() > 1:
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
        bincount = torch.bincount(self.e_idx, minlength=num_p).float().to(self.device)
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p).float().to(self.device) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            self.data_p[:, i] = p_target

        # replace nan by original data
        mask = torch.isnan(self.data_p).any(dim=1)
        self.data_p[mask] = self.data_p_original[mask].clone()
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))
        # return max p coor change
        return True if max_change_pct < 0.01 else False

    def update_p_reg_potential(self, iter_p, reg=0.1):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            reg float: regularizer weight

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

        p0 = torch.zeros_like(self.data_p)
        num_p = self.data_p.shape[0]
        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(self.e_idx, minlength=num_p).float()
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Abort this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            return False
        eps = 1e-8
        for i in range(p0.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        # regularize
        optimizer = optim.SGD([self.data_p], lr=0.05)
        for _ in range(10):
            optimizer.zero_grad()
            loss = f(self.data_p, p0, self.label_p, reg=reg)
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

        p0 = torch.zeros(self.data_p.shape).float().to(self.device)
        num_p = self.data_p.shape[0]
        max_change_pct = 0.0
        # update p to the centroid of its clustered e samples
        bincount = torch.bincount(self.e_idx, minlength=num_p).float()
        if 0 in bincount:
            print('Empty cluster found, optimal transport probably did not converge\n'
                  'Aborting this round of updating'
                  'Try larger lr or max_iter after checking the measures.')
            # return False
        eps = 1e-8
        for i in range(p0.shape[1]):
            # update p to the centroid of their correspondences one dimension at a time
            p_target = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount
            change_pct = torch.max(torch.abs((self.data_p[:, i] - p_target) / (self.data_p[:, i])+eps))
            max_change_pct = max(max_change_pct, change_pct)
            p0[:, i] = p_target
        print("iter {0:d}: max centroid change {1:.2f}%".format(iter_p, 100 * max_change_pct))

        pt = self.data_p.clone().detach()
        pt = utils.estimate_transform_target_pytorch(pt, p0)

        # Replace nan with original data
        mask = torch.isnan(p0).any(dim=1)
        p0[mask] = self.data_p[mask].clone().detach()

        # regularize
        self.data_p = 1 / (1 + reg) * p0 + reg / (1 + reg) * pt

        # return max change
        # return True if max_change_pct < 0.01 else False
        return False



class VotAP:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the area samples

    def __init__(self, data, sampling='unisquare', label=None, weight_p=None, thres=1e-5, ratio=100, verbose=True, device='cpu'):
        """ set up parameters
        Args:
            thres float: threshold to break loops
            ratio float: the ratio of num of e to the num of p
            data pytorch Tensor: initial coordinates of p
            label pytorch Tensor: labels of p
            mass_p pytorch Tensor: weights of p

        Atts:
            thres    float: Threshold to break loops
            lr       float: Learning rate
            verbose   bool: console output verbose flag
            data_p    pytorch floattensor: coordinates of p
            label_p   pytorch inttensor: labels of p
            mass_p    pytorch floattensor: mass of clusters of p
            weight_p   pytorch floattensor: dirac measure of p
        """

        if not isinstance(data, torch.Tensor):
            raise Exception('input is not a pytorch tensor')
        if label and not isinstance(label, torch.Tensor):
            raise Exception('label is neither a numpy array not a pytorch tensor')
        if weight_p and not isinstance(weight_p, torch.Tensor):
            raise Exception('label is neither a numpy array not a pytorch tensor')

        self.data_p = data
        self.data_p_original = self.data_p.clone()
        num_p = data.shape[0]

        self.label_p = label

        self.weight_p = weight_p if weight_p is not None else torch.ones(num_p).double().to(device) / num_p

        self.thres = thres
        self.verbose = verbose
        self.ratio = ratio
        self.device = device

        utils.assert_boundary(self.data_p)

        num_e = int(self.ratio * num_p)
        dim = self.data_p.shape[1]
        self.data_e, _ = utils.random_sample(num_e, dim, sampling=sampling)
        self.data_e = torch.from_numpy(self.data_e).double().to(self.device)

        self.dist = torch.cdist(self.data_p, self.data_e, p=2).double().to(self.device)**2

    def map(self, plot_filename=None, beta=0.9, max_iter=1000, lr=0.5, lr_decay=200, early_stop=200):
        """ map p into the area

        Args:
            plot_filename (string): filename of the gif image
            beta (float): gradient descent momentum
            max_iter (int): maximum number of iteration
            lr (float): learning rate
            lr_decay (int): learning rate decay interval
            early_stop (int): early_stop checking frequency

        :return:
            e_idx (pytorch Tensor): assignment of e to p
            pred_label_e (pytorch Tensor): labels of e that come from nearest p
        """

        num_p = self.data_p.shape[0]
        num_e = self.ratio * num_p

        imgs = []
        dh = 0

        e_idx = None
        running_median, previous_median = [], 0

        for i in range(max_iter):
            # find nearest p for each e
            e_idx = torch.argmin(self.dist, dim=0)

            # calculate total mass of each cell
            mass_p = torch.bincount(e_idx, minlength=num_p).to(self.device) / num_e
            # gradient descent with momentum and decay
            dh = beta * dh + (1-beta) * (mass_p - self.weight_p)
            if i != 0 and i % lr_decay == 0:
                lr *= 0.9
            self.dist += lr * dh[:, None]

            # plot to gif, TODO this is time consuming, got a better way?
            if plot_filename:
                fig = utils.plot_map(self.data_e.cpu().numpy(), e_idx.cpu().numpy() / (num_p - 1))
                img = utils.fig2data(fig)
                imgs.append(img)

            # check if converge
            max_change = torch.max((mass_p - self.weight_p) / self.weight_p)
            if max_change.numel() > 1:
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
                            torch.abs(torch.median(torch.Tensor(running_median)) - previous_median) / previous_median < 0.02:
                        if self.verbose:
                            print("loss saturated, early stopped")
                        break
                    else:
                        previous_median = torch.median(torch.Tensor(running_median))
                        running_median = []

            if max_change <= 1:
                break
        if plot_filename and imgs:
            imageio.mimsave(plot_filename, imgs, fps=4)
        # labels come from centroids
        pred_label_e = self.label_p[e_idx] if self.label_p is not None else None

        # update coordinates of p
        bincount = torch.bincount(e_idx, minlength=num_p).double()
        if 0 in bincount:
            print('Empty cluster found, optimal transport did not converge\nTry larger lr or max_iter')
            # return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = torch.bincount(e_idx, weights=self.data_e[:, i], minlength=num_p) / bincount

        return e_idx, pred_label_e
