# encoding=utf-8
"""
    Created on 10:40 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import FitnessFunction

import GFK


def kernel(ker, X, X2, gamma):
    if not ker or ker == 'primal':
        return X
    elif ker == 'linear':
        if not X2:
            K = np.dot(X.T, X)
        else:
            K = np.dot(X.T, X2)
    elif ker == 'rbf':
        n1sq = np.sum(X ** 2, axis=0)
        n1 = X.shape[1]
        if not X2:
            D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
        else:
            n2sq = np.sum(X2 ** 2, axis=0)
            n2 = X2.shape[1]
            D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X)
        K = np.exp(-gamma * D)
    elif ker == 'sam':
        if not X2:
            D = np.dot(X.T, X)
        else:
            D = np.dot(X.T, X2)
        K = np.exp(-gamma * np.arccos(D) ** 2)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))
    clf = svm.LinearSVC(random_state=0, loss='hinge')
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


class MEDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, rho=1.0, eta=0.1, p=10, gamma=1, T=10, out=None):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf' | 'sam'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param rho: rho in equation
        :param eta: eta in equation
        :param p: number of neighbors
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.rho = rho
        self.eta = eta
        self.gamma = gamma
        self.p = p
        self.T = T
        self.out = out

    def estimate_mu(self, _X1, _Y1, _X2, _Y2):
        adist_m = proxy_a_distance(_X1, _X2)
        C = len(np.unique(_Y1))
        epsilon = 1e-3
        list_adist_c = []
        for i in range(1, C + 1):
            ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
            Xsi = _X1[ind_i[0], :]
            Xtj = _X2[ind_j[0], :]
            adist_i = proxy_a_distance(Xsi, Xtj)
            list_adist_c.append(adist_i)
        adist_c = sum(list_adist_c) / C
        mu = adist_c / (adist_c + adist_m)
        if mu > 1:
            mu = 1
        if mu < epsilon:
            mu = 0
        return mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        gfk = GFK.GFK(dim=self.dim)
        _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        ns, nt = Xs_new.shape[1], Xt_new.shape[1]
        C = len(np.unique(Ys))
        list_acc = []
        YY = np.zeros((ns, C))
        for c in range(1, C + 1):
            ind = np.where(Ys == c)
            YY[ind, c - 1] = 1
        YY = np.vstack((YY, np.zeros((nt, C))))

        X /= np.linalg.norm(X, axis=0)
        L = 0  # Graph Laplacian is on the way...
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(X[:, :ns].T, Ys.ravel())
        Cls = knn_clf.predict(X[:, ns:].T)

        K = kernel(self.kernel_type, X, X2=None, gamma=self.gamma)
        E = np.diagflat(np.vstack((np.ones((ns, 1)), np.zeros((nt, 1)))))

        e = np.vstack((1.0 / ns * np.ones((ns, 1)), -1.0 / nt * np.ones((nt, 1))))
        M0 = e * e.T * C

        for t in range(1, self.T + 1):
            # mu = self.estimate_mu(Xs_new.T, Ys, Xt_new.T, Cls)
            mu = 0.5
            N = 0
            for c in range(1, C + 1):
                e = np.zeros((ns + nt, 1))
                tt = Ys == c
                e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
                yy = Cls == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                if len(Cls[np.where(Cls == c)]) == 0:
                    e[tuple(inds)] = 0.0
                else:
                    e[tuple(inds)] = -1.0 / len(Cls[np.where(Cls == c)])
                N += np.dot(e, e.T)
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            left = np.dot(E + self.lamb * M + self.rho * L, K) + self.eta * np.eye(ns + nt, ns + nt)
            Beta = np.dot(np.linalg.inv(left), np.dot(E, YY))

            # For testing
            Ytest = np.copy(YY)
            for c in range(1, C + 1):
                yy = Cls == c
                inds = np.where(yy == True)
                inds = [item + ns for item in inds]
                Ytest[inds, c - 1] = 1
            SRM = np.linalg.norm(np.dot(Ytest.T - np.dot(Beta.T, K), E)) \
                  + self.eta * np.linalg.multi_dot([Beta.T, K, Beta]).trace()
            MMD = self.lamb * np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([K, M, K]), Beta]).trace()
            fitness = SRM + MMD
            # print(fitness, SRM, MMD)

            F = np.dot(K, Beta)
            Cls = np.argmax(F, axis=1) + 1
            Cls = Cls[ns:]
            acc = np.mean(Cls == Yt.ravel())
            list_acc.append(acc)
            self.out.write("Iteration %d, Fitness %f\n" % (t, fitness))
            # print('MEDA iteration [{}/{}]: mu={:.2f}, Acc={:.4f}'.format(t, self.T, mu, acc))
            # print('=============================================')
        return acc, Cls, list_acc


if __name__ == '__main__':
    datasets = np.array(['GasSensor1-4', 'GasSensor1-2', 'GasSensor1-3',
                         'GasSensor1-5', 'GasSensor1-6', 'GasSensor1-7',
                         'GasSensor1-8', 'GasSensor1-9', 'GasSensor1-10',
                         'SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a',
                         'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c',
                         'SURFd-w', 'SURFw-a', 'SURFw-c', 'SURFw-d',
                         'MNIST-USPS', 'USPS-MNIST'])

    for dataset in datasets:
        source = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Source",
                               delimiter=",")
        m = source.shape[1] - 1
        Xs = source[:, 0:m]
        Ys = np.ravel(source[:, m:m + 1])
        Ys = np.array([int(label) for label in Ys])

        target = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Target",
                               delimiter=",")
        Xt = target[:, 0:m]
        Yt = np.ravel(target[:, m:m + 1])
        Yt = np.array([int(label) for label in Yt])

        file = open("/home/nguyenhoai2/Grid/results/R-MEDA/" + dataset + "/MEDA_iteration.txt", "w")
        meda = MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=100, out=file)
        acc, ypre, list_acc = meda.fit_predict(Xs, Ys, Xt, Yt)
        file.write(str(acc))
        file.close()
