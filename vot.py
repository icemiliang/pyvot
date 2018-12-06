# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Dec 2nd 2018

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class Vot:
    """ variational optimal transportation """

    def setup(self, max_iter_h=1000, max_iter_p=10, thres=1e-8, rate=0.1):
        """ set up parameters

        Args:
            max_iter_h (int): max number of iterations of clustering
            max_iter_p (int): max number of iterations of transportation
            thres (float): threshold to break loops
            rate  (float): learning rate
        """

        self.h = np.zeros(self.np)
        self.thres = thres
        self.learnrate = rate
        self.max_iter_h = max_iter_h
        self.max_iter_p = max_iter_p

    def import_data_file(self, pfilename, efilename, mass=False, label=True):
        """ import data from files

        Args:
            pfilename (string): filename of p
            efilename (string): filename of e
            mass (bool): whether data has a mass column
        """

        p_data = np.loadtxt(open(pfilename, "r"), delimiter=",")
        e_data = np.loadtxt(open(efilename, "r"), delimiter=",")
        self.import_data(p_data, e_data, mass, label)

    def import_data(self, p_data, e_data, mass=False, label=True):
        """ import data from files

        Args:
            p_data np.ndarray(np,dim+): data of p, labels, mass, coordinates, etc
            e_data np.ndarray(ne,dim+): data of e, labels, mass, coordinates, etc
            mass (bool): whether data has a mass column
            label (bool): whether data has a label column
        """

        self.np = np.size(p_data, 0)
        self.ne = np.size(e_data, 0)

        if label and mass:
            self.p_label = p_data[:,0]
            self.e_label = e_data[:,0]
            self.p_dirac = p_data[:,1]
            self.e_mass = e_data[:,1]
            self.p_coor = p_data[:,2:]
            self.e_coor = e_data[:,2:]
        elif label and not mass:
            self.p_label = p_data[:,0]
            self.e_label = e_data[:,0]
            self.p_dirac = np.ones(self.np)/self.np
            self.e_mass = np.ones(self.ne)/self.ne
            self.p_coor = p_data[:,1:]
            self.e_coor = e_data[:,1:]
        elif not label and mass:
            self.p_label = -np.ones(self.ne)
            self.e_label = -np.ones(self.np)
            self.p_dirac = p_data[:, 1]
            self.e_mass = e_data[:, 1]
            self.p_coor = p_data[:, 1:]
            self.e_coor = e_data[:, 1:]
        else: # not label and not mass
            self.p_label = -np.ones(self.ne)
            self.e_label = -np.ones(self.np)
            self.p_dirac = np.ones(self.np)/self.np
            self.e_mass = np.ones(self.ne)/self.ne
            self.p_coor = p_data
            self.e_coor = e_data

        self.p_mass = np.zeros(self.np) # mass is the sum of its e's weights, its own weight is "dirac"
        self.p_label = self.p_label.astype(int)
        self.e_label = self.e_label.astype(int)

    def cluster(self, reg=0, alpha=0.01):
        """ compute Wasserstein clustering

        Args:
            reg int: flag for regularization, 0 means no regularization

        See Also
        --------
        update_p : update p
        update_map: computer optimal transportation
        """

        for iter_p in range(self.max_iter_p):
            self.cost_base = cdist(self.p_coor, self.e_coor, 'sqeuclidean')
            for iter_h in range(self.max_iter_h):
                if self.update_map(iter_p,iter_h): break
            if self.update_p(iter_p, reg, alpha): break

    def update_map(self, iter_p, iter_h):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index of clustering
            iter_h int: iteration index of transportation

        Returns:
            float: max change of derivative, small max means convergence
        """

        # update dist matrix
        self.cost = self.cost_base - self.h[:, np.newaxis]
        # find nearest p for each e and add mass to p
        self.e_idx = np.argmin(self.cost, axis=0)
        # labels come from centroids
        self.e_predict = self.p_label[self.e_idx]
        for j in range(self.np):
            self.p_mass[j] = np.sum(self.e_mass[self.e_idx == j])
        # update gradient and h
        grad = self.p_mass - self.p_dirac
        self.h = self.h - self.learnrate * grad
        # check if converge and return max derivative
        return True if np.amax(grad) < self.thres else False

    def update_p(self, iter_p, reg=0, alpha=0.01):
        """ update p

        Args:
            iter_p int: iteration index
            reg int or string: regularizer type
            alpha float: regularizer weight

        Returns:
            float: max change of p, small max means convergence
        """

        if reg == 1 or reg == 'potential':
            return self.update_p_reg_potential(iter_p, alpha)
        elif reg == 2 or reg == 'curvature':
            return self.update_p_reg_curvature(iter_p)
        else:
            return self.update_p_noreg(iter_p)

    def update_p_noreg(self, iter_p):
        """ update each p to the centroids of its cluster

        Args:
            iter_p int: iteration index

        Returns:
            float: max change of p, small max means convergence
        """

        max_change = 0.0
        # update p to the centroid of its clustered e samples
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.np):
            weight = self.e_mass[self.e_idx == j]
            if weight.size == 0:
                continue
            p_target = np.average(self.e_coor[self.e_idx == j,:], weights=weight, axis=0)
            # check if converge
            max_change = max(np.amax(self.p_coor[j,:] - p_target), max_change)
            self.p_coor[j,:] = p_target
        print("iter " + str(iter_p) + ": " + str(max_change))
        # return max p coor change
        return True if max_change < self.thres else False

    def update_p_reg_potential(self, iter_p, alpha=0.01):
        """ update each p to the centroids of its cluster,
            regularized by intra-class distances

        Args:
            iter_p int: index of the iteration of updating p
            alpha float: regularizer weight

        Returns:
            float: max change of p, small max means convergence
        """

        def f(p, p0, label=None, alpha=0.01):
            """ objective function incorporating labels

            Args:
                p  np.array(np,dim):   p
                p0 np.array(np,dim):  centroids of e
                label np.array(np,): labels of p
                alpha float: regularizer weight

            Returns:
                float: f = sum(|p-p0|^2) + alpha*sum(delta*|pi-pj|^2), delta = 1 if li == lj
            """

            p = p.reshape(p0.shape)
            reg_term = 0.0
            # TODO Replace the nested for loop with matrix/vector operations, if possible
            for idx, l in np.ndenumerate(np.unique(label)):
                p_sub = p[label == l,:]
                for shift in range(1,np.size(p_sub,0)):
                    reg_term += np.sum(np.sum((p_sub-np.roll(p_sub, shift, axis=0))**2.0))

            return np.sum(np.sum((p - p0)**2.0)) + alpha*reg_term

        max_change = 0.0
        p0 = np.zeros((self.p_coor.shape))
        # new controid pos
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.np):
            p0[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - p0[j,:]),max_change)
        print("iter " + str(iter_p) + ": " + str(max_change))
        # regularize
        res = minimize(f, self.p_coor, method='BFGS', tol=self.thres, args=(p0,self.p_label,alpha))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(p0.shape)
        # return max change
        return True if max_change < self.thres else False

    def update_p_reg_curvature(self, iter_p, alpha1=0.01, alpha2=0.01):
        """ update each p to the centroids of its cluster,
            regularized by length and curvature

            Note that some variables should be modified to fit the specific skeleton

        Args:
            iter_p int: index of the iteration of updating p
            alpha1 float: length term
            alpha2 float: curvature term

        Returns:
            float: max change of p, small max means convergence
        """

        def f(p, p0, pfix, alpha1=0.01, alpha2=0.1):
            """ objective function incorporating length and curvature

            Args:
                p  np.ndarray(np-pfix_num,dim): p
                p0 np.ndarray(np-pfix_num,dim): centroids of e
                pfix np.ndarray(fix_num,): labels of p
                alpha1 float: weight of regularizer length
                alpha2 float: weight of regularizer curvature

            Returns:
                float: f = sum(|p-p0|^2) + alpha1*sum(length(pj,pj-1) + alpha2*sum(curvature(pj,pj-1,pj-2)))
            """

            def length(p1, p2, p3):
                if p1.ndim == 1:
                    p1 = p1.reshape((1,-1)); p2 = p2.reshape((1,-1)); p3 = p3.reshape((1,-1))
                x1 = p1[:,0]; y1 = p1[:,1]; z1 = p1[:,2]
                x2 = p2[:,0]; y2 = p2[:,1]; z2 = p2[:,2]
                x3 = p3[:,0]; y3 = p3[:,1]; z3 = p3[:,2]

                a = x1 - 2*x2 + x3
                b = y1 - 2*y2 + y3
                c = z1 - 2*z2 + z3
                dx = x1 - x2
                dy = y1 - y2
                dz = z1 - z2

                t = np.arange(0.0, 1.01, 0.01)
                ds = np.sqrt((a*t - dx)**2 + (b*t - dy)**2 + (c*t - dz)**2)
                ds[0] /= 2
                ds[100] /= 2
                return np.sum(ds)/100.0

            def curvature(p1, p2, p3):
                if p1.ndim == 1:
                    p1 = p1.reshape((1,-1)); p2 = p2.reshape((1,-1)); p3 = p3.reshape((1,-1))
                x1 = p1[:,0]; y1 = p1[:,1]; z1 = p1[:,2]
                x2 = p2[:,0]; y2 = p2[:,1]; z2 = p2[:,2]
                x3 = p3[:,0]; y3 = p3[:,1]; z3 = p3[:,2]

                a = x1 - 2*x2 + x3
                b = y1 - 2*y2 + y3
                c = z1 - 2*z2 + z3
                dx = x1 - x2
                dy = y1 - y2
                dz = z1 - z2
                numerator = (c*dy - b*dz)**2 + (c*dx - a*dz)**2 + (b*dx - a*dy)**2

                t = np.arange(0.0, 1.01, 0.01)
                k = numerator / np.power((a*t - dx)**2 + (b*t - dy)**2 + (c*t - dz)**2, 2.5)
                k[0] /= 2
                k[100] /= 2
                return np.sum(k)/100.0

            # The following block of code should be modified to fit specific skeletons

            cost_length = 0.0
            cost_curvature = 0.0
            pfix = pfix.reshape((-1,3))
            p = p.reshape((-1, 3))
            p0 = p0.reshape((-1, 3))

            # return 0.05*np.sum(np.sum((p0 - p)**2.0)) + \
            #        length(pfix[0,:], p[1,:], p[2,:]) + \
            #        length(p[1, :], p[2, :], p[3, :]) + \
            #        length(p[2, :], p[3, :], p[4, :]) + \
            #        length(p[3, :], p[4, :], p[5, :]) + \
            #        length(p[4, :], p[5, :], p[6, :]) + \
            #        length(p[5, :], p[6, :], p[7, :]) + \
            #        length(p[6, :], p[7, :], p[8, :]) + \
            #        length(p[7, :], p[8, :], pfix[3, :]) + \
            #        curvature(pfix[0, :], p[1, :], p[2, :]) + \
            #        curvature(p[1, :], p[2, :], p[3, :]) + \
            #        curvature(p[2, :], p[3, :], p[4, :]) + \
            #        curvature(p[3, :], p[4, :], p[5, :]) + \
            #        curvature(p[4, :], p[5, :], p[6, :]) + \
            #        curvature(p[5, :], p[6, :], p[7, :]) + \
            #        curvature(p[6, :], p[7, :], p[8, :]) + \
            #        curvature(p[7, :], p[8, :], pfix[3, :]) + \
            #        length(pfix[1, :], p[11, :], p[12, :]) + \
            #        length(p[11, :], p[12, :], p[13, :]) + \
            #        length(p[12, :], p[13, :], p[14, :]) + \
            #        length(p[13, :], p[14, :], p[15, :]) + \
            #        length(p[14, :], p[15, :], p[16, :]) + \
            #        length(p[15, :], p[16, :], p[17, :]) + \
            #        length(p[16, :], p[17, :], p[18, :]) + \
            #        length(p[17, :], p[18, :], pfix[3, :]) + \
            #        curvature(pfix[1, :], p[11, :], p[12, :]) + \
            #        curvature(p[11, :], p[12, :], p[13, :]) + \
            #        curvature(p[12, :], p[13, :], p[14, :]) + \
            #        curvature(p[13, :], p[14, :], p[15, :]) + \
            #        curvature(p[14, :], p[15, :], p[16, :]) + \
            #        curvature(p[15, :], p[16, :], p[17, :]) + \
            #        curvature(p[16, :], p[17, :], p[18, :]) + \
            #        curvature(p[17, :], p[18, :], pfix[3, :]) + \
            #        length(pfix[2, :], p[20, :], p[21, :]) + \
            #        length(p[20, :], p[21, :], pfix[3, :]) + \
            #        curvature(pfix[2, :], p[20, :], p[21, :]) + \
            #        curvature(p[20, :], p[21, :], pfix[3, :]) + \
            #        curvature(p[18, :], pfix[3, :], p[21, :])


            # tmp = curvature(p[1, :], p[2, :], p[3, :])
            # print(p[1, :], end="")
            # print(p[2, :], end="")
            # print(p[3, :], end="")
            # print(tmp)

            cost_curvature = curvature(pfix[0, :], p[1, :], p[2, :]) + \
                   curvature(p[1, :], p[2, :], p[3, :]) + \
                   curvature(p[2, :], p[3, :], p[4, :]) + \
                   curvature(p[3, :], p[4, :], p[5, :]) + \
                   curvature(p[4, :], p[5, :], p[6, :]) + \
                   curvature(p[5, :], p[6, :], p[7, :]) + \
                   curvature(p[6, :], p[7, :], p[8, :]) + \
                   curvature(p[7, :], p[8, :], pfix[3, :]) + \
                   curvature(pfix[1, :], p[11, :], p[12, :]) + \
                   curvature(p[11, :], p[12, :], p[13, :]) + \
                   curvature(p[12, :], p[13, :], p[14, :]) + \
                   curvature(p[13, :], p[14, :], p[15, :]) + \
                   curvature(p[14, :], p[15, :], p[16, :]) + \
                   curvature(p[15, :], p[16, :], p[17, :]) + \
                   curvature(p[16, :], p[17, :], p[18, :]) + \
                   curvature(p[17, :], p[18, :], pfix[3, :]) + \
                   curvature(pfix[2, :], p[20, :], p[21, :]) + \
                   curvature(p[20, :], p[21, :], pfix[3, :]) + \
                   curvature(p[18, :], pfix[3, :], p[21, :])
            cost = np.sum(np.sum((p0 - p)**2.0))
            print("data term: " + str(cost_curvature))
            print("curvature: " + str(cost))
            return 0.01*np.sum(np.sum((p0 - p)**2.0)) + \
                   curvature(pfix[0, :], p[1, :], p[2, :]) + \
                   curvature(p[1, :], p[2, :], p[3, :]) + \
                   curvature(p[2, :], p[3, :], p[4, :]) + \
                   curvature(p[3, :], p[4, :], p[5, :]) + \
                   curvature(p[4, :], p[5, :], p[6, :]) + \
                   curvature(p[5, :], p[6, :], p[7, :]) + \
                   curvature(p[6, :], p[7, :], p[8, :]) + \
                   curvature(p[7, :], p[8, :], pfix[3, :]) + \
                   curvature(pfix[1, :], p[11, :], p[12, :]) + \
                   curvature(p[11, :], p[12, :], p[13, :]) + \
                   curvature(p[12, :], p[13, :], p[14, :]) + \
                   curvature(p[13, :], p[14, :], p[15, :]) + \
                   curvature(p[14, :], p[15, :], p[16, :]) + \
                   curvature(p[15, :], p[16, :], p[17, :]) + \
                   curvature(p[16, :], p[17, :], p[18, :]) + \
                   curvature(p[17, :], p[18, :], pfix[3, :]) + \
                   curvature(pfix[2, :], p[20, :], p[21, :]) + \
                   curvature(p[20, :], p[21, :], pfix[3, :]) + \
                   curvature(p[18, :], pfix[3, :], p[21, :])

        # find nearest p for each e and add mass to p
        idx_tmp = np.argmin(self.cost_base, axis=0)
        dirac_tmp = np.zeros(self.np)
        for j in range(self.np):
            dirac_tmp[j] = np.sum(self.e_mass[idx_tmp == j])

        self.p_dirac = 0.9 * self.p_dirac + 0.1 * dirac_tmp

        pfix = np.concatenate((self.p_coor[0,:],self.p_coor[10,:],self.p_coor[19,:],self.p_coor[9,:]), axis=0)

        max_change = 0.0
        p0 = np.zeros((self.p_coor.shape))
        # new controid pos
        # TODO Replace the for loop with matrix/vector operations, if possible
        for j in range(self.np):
            p0[j,:] = np.average(self.e_coor[self.e_idx == j,:], weights=self.e_mass[self.e_idx == j], axis=0)
            max_change = max(np.amax(self.p_coor[j,:] - p0[j,:]),max_change)
        print("iter " + str(iter_p) + ": " + str(max_change))
        res = minimize(f, self.p_coor, method='BFGS', tol=self.thres, args=(p0, pfix, alpha1, alpha2))
        self.p_coor = res.x
        self.p_coor = self.p_coor.reshape(p0.shape)

        pfix = pfix.reshape((-1, 3))
        self.p_coor[0,:] = pfix[0,:]
        self.p_coor[10,:] = pfix[1,:]
        self.p_coor[19,:] = pfix[2,:]
        self.p_coor[9, :] = pfix[3, :]

        # return max change
        return True if max_change < self.thres else False
