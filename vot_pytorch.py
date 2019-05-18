# PyVot
# Variational Wasserstein Clustering
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 15th 2019


from skimage import transform as tf
import imageio
import utils
import torch


class VotAP:
    """ Area Preserving with variational optimal transportation """
    # p are the centroids
    # e are the area samples

    def __init__(self, data, label=None, mass_p=None, thres=1e-5, ratio=100, verbose=True, device='cpu'):
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
        num_p = self.data_p.shape[0]
        num_e = self.ratio * num_p
        dim = self.data_p.shape[1]
        self.data_e, _ = utils.random_sample(num_e, dim, sampling=sampling, has_label=self.label_p)
        self.data_e = torch.from_numpy(self.data_e).float().to(self.device)

        base_dist = torch.cdist(self.data_p, self.data_e, p=2)
        self.e_idx = torch.argmin(base_dist, dim=0)
        h = torch.zeros(num_p).float().to(self.device)
        imgs = []
        dh = 0

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

            # check if converge and return max derivative
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
            return
        for i in range(self.data_p.shape[1]):
            # update p to the centroid of their correspondences
            self.data_p[:, i] = torch.bincount(self.e_idx, weights=self.data_e[:, i], minlength=num_p).float() / bincount
