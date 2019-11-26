# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import Helpers as Pre


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1.0, gamma=1.0):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
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
    dims = [60, 70, 80]
    lambdas = [0.1, 0.1, 10.0]
    normalizes = [0, 1, 0]

    for index, datasets in enumerate(list_datasets):
        dim = dims[index]
        lamb = lambdas[index]
        normalize = normalizes[index]
        for dataset in datasets:
            print(dataset)
            dir = '/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/' + dataset + '/'

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

            tca = TCA(kernel_type='rbf', dim=dim, lamb=lamb, gamma=0.5)
            acc, _ = tca.fit_predict(Xs, Ys, Xt, Yt)

            file = open('/home/nguyenhoai2/Grid/results/MEDA-Extend/' + dataset + '/TCA', 'w')
            file.write(str(acc))
            file.close()