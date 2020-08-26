# encoding=utf-8
"""
    Created on 17:25 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import bob.learn
import bob.learn.linear
import bob.math
from sklearn.neighbors import KNeighborsClassifier
import Helpers as Pre
import os


class GFK:
    def __init__(self, dim=20):
        '''
        Init func
        :param dim: dimension after GFK
        '''
        self.dim = dim
        self.eps = 1e-20

    def fit(self, Xs, Xt, norm_inputs=None):
        '''
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        '''
        if norm_inputs:
            source, mu_source, std_source = self.znorm(Xs)
            target, mu_target, std_target = self.znorm(Xt)
        else:
            mu_source = np.zeros(shape=(Xs.shape[1]))
            std_source = np.ones(shape=(Xs.shape[1]))
            mu_target = np.zeros(shape=(Xt.shape[1]))
            std_target = np.ones(shape=(Xt.shape[1]))
            source = Xs
            target = Xt

        Ps = self.train_pca(source, mu_source, std_source, 0.99)
        Pt = self.train_pca(target, mu_target, std_target, 0.99)
        Ps = np.hstack((Ps.weights, scipy.linalg.null_space(Ps.weights.T)))
        Pt = Pt.weights[:, :self.dim]
        N = Ps.shape[1]
        dim = Pt.shape[1]

        # Principal angles between subspaces
        QPt = np.dot(Ps.T, Pt)

        # [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
        A = QPt[0:dim, :].copy()
        B = QPt[dim:, :].copy()

        # Equation (2)
        [V1, V2, V, Gam, Sig] = bob.math.gsvd(A, B)
        V2 = -V2

        # Some sanity checks with the GSVD
        I = np.eye(V1.shape[1])
        I_check = np.dot(Gam.T, Gam) + np.dot(Sig.T, Sig)
        assert np.sum(abs(I - I_check)) < 1e-10

        theta = np.arccos(np.diagonal(Gam))

        # Equation (6)
        B1 = np.diag(0.5 * (1 + (np.sin(2 * theta) / (2. * np.maximum
        (theta, 1e-20)))))
        B2 = np.diag(0.5 * ((np.cos(2 * theta) - 1) / (2 * np.maximum(
            theta, self.eps))))
        B3 = B2
        B4 = np.diag(0.5 * (1 - (np.sin(2 * theta) / (2. * np.maximum
        (theta, self.eps)))))

        # Equation (9) of the suplementary matetial
        delta1_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta1_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta1 = np.vstack((delta1_1, delta1_2))

        delta2_1 = np.hstack((B1, B2, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_2 = np.hstack((B3, B4, np.zeros(shape=(dim, N - 2 * dim))))
        delta2_3 = np.zeros(shape=(N - 2 * dim, N))
        delta2 = np.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta3_2 = np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta3 = np.vstack((delta3_1, delta3_2)).T

        delta = np.dot(np.dot(delta1, delta2), delta3)
        G = np.dot(np.dot(Ps, delta), Ps.T)
        sqG = scipy.real(scipy.linalg.fractional_matrix_power(G, 0.5))
        Xs_new, Xt_new = np.dot(sqG, Xs.T).T, np.dot(sqG, Xt.T).T
        return G, Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Fit and use 1NN to classify
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy, predicted labels of target domain, and G
        '''
        G, Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = np.mean(y_pred == Yt.ravel())
        return acc, y_pred, G

    def principal_angles(self, Ps, Pt):
        """
        Compute the principal angles between source (:math:`P_s`) and target (:math:`P_t`) subspaces in a Grassman which is defined as the following:

        :math:`d^{2}(P_s, P_t) = \sum_{i}( \theta_i^{2} )`,

        """
        # S = cos(theta_1, theta_2, ..., theta_n)
        _, S, _ = np.linalg.svd(np.dot(Ps.T, Pt))
        thetas_squared = np.arccos(S) ** 2

        return np.sum(thetas_squared)

    def train_pca(self, data, mu_data, std_data, subspace_dim):
        '''
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param mu_data: mu
        :param std_data: std
        :param subspace_dim: dim
        :return: a wrapped machine object
        '''
        t = bob.learn.linear.PCATrainer()
        machine, variances = t.train(data)

        # For re-shaping, we need to copy...
        variances = variances.copy()

        # compute variance percentage, if desired
        if isinstance(subspace_dim, float):
            cummulated = np.cumsum(variances) / np.sum(variances)
            for index in range(len(cummulated)):
                if cummulated[index] > subspace_dim:
                    subspace_dim = index
                    break
            subspace_dim = index
        machine.resize(machine.shape[0], subspace_dim)
        machine.input_subtract = mu_data
        machine.input_divide = std_data

        return machine

    def znorm(self, data):
        """
        Z-Normaliza
        """
        mu = np.average(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mu) / std
        return data, mu, std

    def subspace_disagreement_measure(self, Ps, Pt, Pst):
        """
        Get the best value for the number of subspaces
        For more details, read section 3.4 of the paper.
        **Parameters**
          Ps: Source subspace
          Pt: Target subspace
          Pst: Source + Target subspace
        """

        def compute_angles(A, B):
            _, S, _ = np.linalg.svd(np.dot(A.T, B))
            S[np.where(np.isclose(S, 1, atol=self.eps) == True)[0]] = 1
            return np.arccos(S)

        max_d = min(Ps.shape[1], Pt.shape[1], Pst.shape[1])
        alpha_d = compute_angles(Ps, Pst)
        beta_d = compute_angles(Pt, Pst)
        d = 0.5 * (np.sin(alpha_d) + np.sin(beta_d))
        return np.argmax(d)


if __name__ == '__main__':
    # import sys, os
    # random_seed = 1617 * 1
    # np.random.seed(random_seed)
    #
    # dataset = sys.argv[1]
    # dim = int(sys.argv[2])
    # norm = int(sys.argv[3]) == 1
    #
    # source = np.genfromtxt("Source", delimiter=",")
    # m = source.shape[1] - 1
    # Xs = source[:, 0:m]
    # Ys = np.ravel(source[:, m:m + 1])
    # Ys = np.array([int(label) for label in Ys])
    #
    # target = np.genfromtxt("Target", delimiter=",")
    # Xt = target[:, 0:m]
    # Yt = np.ravel(target[:, m:m + 1])
    # Yt = np.array([int(label) for label in Yt])
    #
    # if norm:
    #     Xs, Xt = Pre.normalize_data(Xs, Xt)
    #
    # C = len(np.unique(Ys))
    # if C > np.max(Ys):
    #     Ys = Ys + 1
    #     Yt = Yt + 1
    #
    # # Create target Directory if don't exist
    # file = open((str(dim)+'.txt'), "w")
    #
    # gfk = GFK(dim=dim)
    # acc, _, _ = gfk.fit_predict(Xs, Ys, Xt, Yt)
    # file.write(str(acc))
    #
    # file.close()

    list_datasets = [
        ['SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a', 'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c', 'SURFd-w', 'SURFw-a',
         'SURFw-c', 'SURFw-d'],
        # ['DECAF6d-w','DECAF6a-c','DECAF6a-d','DECAF6a-w','DECAF6c-a','DECAF6c-d','DECAF6c-w','DECAF6d-a','DECAF6d-c','DECAF6w-a','DECAF6w-c','DECAF6w-d'],
        # ['ICLEFc-i','ICLEFc-p','ICLEFi-c','ICLEFi-p','ICLEFp-c','ICLEFp-i'],
        ['Office31amazon-dslr', 'Office31amazon-webcam', 'Office31dslr-amazon', 'Office31dslr-webcam',
         'Office31webcam-amazon', 'Office31webcam-dslr'],
        # ['OfficeHomeArt-Clipart','OfficeHomeArt-Product','OfficeHomeArt-RealWorld','OfficeHomeClipart-Art',
        #  'OfficeHomeClipart-Product',
        #         'OfficeHomeClipart-RealWorld', 'OfficeHomeProduct-Art','OfficeHomeProduct-Clipart','OfficeHomeProduct-RealWorld','OfficeHomeRealWorld-Art',
        #         'OfficeHomeRealWorld-Clipart',
        #  'OfficeHomeRealWorld-Product'
        #  ],
        ['AMZbooks-dvd', 'AMZbooks-elec', 'AMZbooks-kitchen', 'AMZdvd-books', 'AMZdvd-elec', 'AMZdvd-kitchen',
         'AMZelec-books', 'AMZelec-dvd',
         'AMZelec-kitchen', 'AMZkitchen-books', 'AMZkitchen-dvd', 'AMZkitchen-elec'],
        # ['Caltech101-ImageNet', 'Caltech101-SUN09', 'ImageNet-Caltech101', 'ImageNet-SUN09', 'SUN09-ImageNet', 'SUN09-Caltech101']
        # ['VOC2007-ImageNet', 'ImageNet-VOC2007']
    ]
    list_datasets = [['SURFd-w']]
    dims = [40, 30, 40]
    normalizes = [0, 1, 0]

    for index, datasets in enumerate(list_datasets):
        dim = dims[index]
        normalize = normalizes[index]
        for dataset in datasets:
            print(dataset)
            dir = '/Volumes/Data/Work/Research/Current/Datasets/Transferlearning/UnPairs/' + dataset + '/'

            source = np.genfromtxt(dir + 'Source', delimiter=',')
            m = source.shape[1] - 1
            Xs = source[:, 0:m]
            Ys = np.ravel(source[:, m:m + 1])
            Ys = np.array([int(label) for label in Ys])

            target = np.genfromtxt(dir + 'Target', delimiter=',')
            Xt = target[:, 0:m]
            Yt = np.ravel(target[:, m:m + 1])
            Yt = np.array([int(label) for label in Yt])

            if normalize:
                Xs, Xt = Pre.normalize_data(Xs, Xt)

            C = len(np.unique(Ys))
            if C > np.max(Ys):
                Ys = Ys + 1
                Yt = Yt + 1

            gfk = GFK(dim=dim)
            acc, _, _ = gfk.fit_predict(Xs, Ys, Xt, Yt)

            # file = open('/home/nguyenhoai2/Grid/results/MEDA-Extend/' + dataset + '/GFK', 'w')
            file = open('GFK', 'w')
            file.write(str(acc))
            file.close()