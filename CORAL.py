# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.svm import LinearSVC as LSVM
import Helpers as Pre


class CORAL:

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral).astype(float)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = LSVM()
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
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

    for index, datasets in enumerate(list_datasets):
        for dataset in datasets:
            dir = '/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/'+dataset+'/'

            source = np.genfromtxt(dir+'Source', delimiter=',')
            m = source.shape[1] - 1
            Xs = source[:, 0:m]
            Ys = np.ravel(source[:, m:m + 1])
            Ys = np.array([int(label) for label in Ys])

            target = np.genfromtxt(dir+'Target', delimiter=',')
            Xt = target[:, 0:m]
            Yt = np.ravel(target[:, m:m + 1])
            Yt = np.array([int(label) for label in Yt])

            if index == 1:
                Xs, Xt = Pre.normalize_data(Xs, Xt)

            C = len(np.unique(Ys))
            if C > np.max(Ys):
                Ys = Ys + 1
                Yt = Yt + 1

            coral = CORAL()
            acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)

            file = open('/home/nguyenhoai2/Grid/results/MEDA-Extend/'+dataset+'/Coral', 'w')
            file.write(str(acc))
