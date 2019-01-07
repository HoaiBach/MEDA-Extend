# encoding=utf-8
"""
    Created on 10:40 2018/12/14
    @author: Bach Nguyen
"""

import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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


class Random_MEDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, rho=1.0, eta=0.1, p=10, gamma=1.0,
                 T=10, init_op=0, re_init_op=0, seed=1617):
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

        self.Xs = None
        self.Ys = None
        self.Xt = None
        self.Yt = None

        self.ns = 0
        self.nt = 0
        self.C = 0

        self.M0 = None
        self.A = None
        self.K = None
        self.YY = None  # 1 hot coding
        self.L = 0
        # self.Yt_pseu = None


        self.init_op = init_op  # 0-random, 1- KNN (diff K), 2-10 diff classifiers
        self.re_init_op = re_init_op # 0-random, 1-using best

        np.random.seed(seed)

    def evolve(self, Xs, Ys, Xt, Yt):
        self.Xs = Xs
        self.Ys = Ys
        self.Xt = Xt
        self.Yt = Yt

        self.ns, self.nt = Xs.shape[0], Xt.shape[0]
        self.C = len(np.unique(Ys))

        # Transform data using gfk
        # should be done faster by reading from gfk file.
        gfk = GFK.GFK(dim=self.dim)
        _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        X /= np.linalg.norm(X, axis=0)
        self.Xs = X[:, :self.ns].T
        self.Xt = X[:, self.ns:].T

        # build some matrices that are not changed
        self.K = kernel(self.kernel_type, X, X2=None, gamma=self.gamma)
        self.A = np.diagflat(np.vstack((np.ones((self.ns, 1)), np.zeros((self.nt, 1)))))
        e = np.vstack((1.0 / self.ns * np.ones((self.ns, 1)), -1.0 / self.nt * np.ones((self.nt, 1))))
        self.M0 = e * e.T * self.C

        self.YY = np.zeros((self.ns, self.C))
        for c in range(1, self.C + 1):
            ind = np.where(self.Ys == c)
            self.YY[ind, c - 1] = 1
        self.YY = np.vstack((self.YY, np.zeros((self.nt, self.C))))

        N = 10
        GEN = 10
        pos_min = -10
        pos_max = 10
        pop = []
        pop_mmd = [sys.float_info.max] * N
        pop_srm = [sys.float_info.max] * N
        pop_fit = [sys.float_info.max] * N
        pop_src_acc = [sys.float_info.max] * N
        pop_tar_acc = [sys.float_info.max] * N
        NBIT = (self.ns + self.nt) * self.C

        # initialization
        if self.init_op == 0:
            # start randomly
            for i in range(N):
                poistion = np.random.uniform(pos_min, pos_max, NBIT)
                beta = np.reshape(poistion, (self.ns + self.nt, self.C))
                pop.append(beta)
        elif self.init_op == 1:
            # using different KNN
            for i in range(N):
                classifier = KNeighborsClassifier(2*i+1)
                beta = self.initialize_with_class(classifier)
                pop.append(beta)
        elif self.init_op == 2:
            # using differnet classifiers
            classifiers = list([])
            classifiers.append(KNeighborsClassifier(1))
            classifiers.append(KNeighborsClassifier(5))
            classifiers.append(SVC(kernel="linear", C=0.025, random_state=1617))
            classifiers.append(SVC(kernel="rbf", C=1, gamma=2, random_state=1617))
            classifiers.append(GaussianProcessClassifier(1.0 * RBF(1.0)))
            classifiers.append(GaussianNB())
            classifiers.append(DecisionTreeClassifier(max_depth=5))
            classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10))
            classifiers.append(AdaBoostClassifier())
            classifiers.append(QuadraticDiscriminantAnalysis())
            assert len(classifiers) == N
            for i in range(N):
                beta = self.initialize_with_class(classifiers[i])
                pop.append(beta)
        else:
            print("Unsupported Initialize Strategy")
            sys.exit(1)

        # evolution
        archive = []
        archive_fit = []
        archive_src_acc = []
        archive_tar_acc = []
        archive_mmd = []
        archive_srm = []
        best = None
        best_fitness = sys.float_info.max
        best_src_acc = -sys.float_info.max
        best_tar_acc = -sys.float_info.max
        best_mmd = 0
        best_srm = 0

        for g in range(GEN):
            print('==============Gen %d===============' % g)
            for index, ind in enumerate(pop):
                # refine the position using gradient descent
                new_position, fitness, mmd, srm, src_acc, tar_acc = self.fit_predict(pop[index])
                print("Ind %d has fitness of %f and source accuracy %f and target accuracy %f."
                      % (index, fitness, src_acc, tar_acc))

                # if the fitness is not improved, store the previous position in the archive
                # randomly create a new position based on the best
                # store the current position to archive
                if pop_fit[index] <= fitness:
                    print("***Reset ind %d" % index)
                    new_position = self.re_initialize(pop, best, pos_min, pos_max, strategy=self.re_init_op)
                    new_position, fitness, mmd, srm, src_acc, tar_acc = self.fit_predict(new_position)
                    print("Ind %d is re-intialized and fitness of %f and source accuracy %f and target accuracy %f.***"
                          % (index, fitness, src_acc, tar_acc))

                    # append the old ind
                    archive.append(pop[index])
                    archive_fit.append(fitness)
                    archive_mmd.append(mmd)
                    archive_srm.append(srm)
                    archive_src_acc.append(src_acc)
                    archive_tar_acc.append(tar_acc)

                # now update the new position with its fitness
                pop[index] = new_position
                pop_fit[index] = fitness
                pop_mmd[index] = mmd
                pop_srm[index] = srm
                pop_src_acc[index] = src_acc
                pop_tar_acc[index] = tar_acc

                # update best if necessary
                if fitness < best_fitness:
                    best = np.copy(pop[index])
                    best_fitness = fitness
                    best_src_acc = src_acc
                    best_tar_acc = tar_acc
                    best_srm = srm
                    best_mmd = mmd

            print("Best fitness of %f and source accuracy %f and target accuracy %f." % (
                best_fitness, best_src_acc, best_tar_acc))

        archive.append(best)
        archive_fit.append(best_fitness)
        archive_mmd.append(best_mmd)
        archive_srm.append(best_srm)
        archive_src_acc.append(best_src_acc)
        archive_tar_acc.append(best_tar_acc)

        nd_indices = self.get_non_dominated(archive, archive_srm, archive_mmd)
        print("================All archive==================")
        for index, ind in enumerate(archive):
            print("Member %d has fitness = %f, srm = %f, mmd = %f, src_acc = %f, tar_acc = %f"
                  % (index, archive_fit[index], archive_srm[index],
                     archive_mmd[index], archive_src_acc[index],
                     archive_tar_acc[index]))
        print("================Non-dominated==================")
        for index in nd_indices:
            print("Member %d has fitness = %f, srm = %f, mmd = %f, src_acc = %f, tar_acc = %f"
                  % (index, archive_fit[index], archive_srm[index],
                     archive_mmd[index], archive_src_acc[index],
                     archive_tar_acc[index]))

        print("Final archive size %d" % len(archive))

        all_labels = []
        print("========From all archive========")
        all_indices = range(len(archive))
        class_indices = all_indices
        for index in class_indices:
            ind = archive[index]
            print("Member %d has fitness = %f, srm = %f, mmd = %f, src_acc = %f, tar_acc = %f"
                  % (index, archive_fit[index], archive_srm[index],
                     archive_mmd[index], archive_src_acc[index],
                     archive_tar_acc[index]))
            F = np.dot(self.K, ind)
            Y_pseudo = np.argmax(F, axis=1) + 1
            Yt_pseu = Y_pseudo[self.ns:].tolist()
            all_labels.append(Yt_pseu)
        all_labels = np.array(all_labels)
        vote_label = []
        for ins_idx in range(all_labels.shape[1]):
            ins_labels = all_labels[:, ins_idx]
            counts = np.bincount(ins_labels)
            label = np.argmax(counts)
            vote_label.append(label)
        acc = np.mean(vote_label == Yt)
        print(acc)

        all_labels = []
        print("========From non-dominated========")
        class_indices = nd_indices
        for index in class_indices:
            ind = archive[index]
            print("Member %d has fitness = %f, srm = %f, mmd = %f, src_acc = %f, tar_acc = %f"
              % (index, archive_fit[index], archive_srm[index],
                 archive_mmd[index], archive_src_acc[index],
                 archive_tar_acc[index]))
            F = np.dot(self.K, ind)
            Y_pseudo = np.argmax(F, axis=1) + 1
            Yt_pseu = Y_pseudo[self.ns:].tolist()
            all_labels.append(Yt_pseu)
        all_labels = np.array(all_labels)
        vote_label = []
        for ins_idx in range(all_labels.shape[1]):
            ins_labels = all_labels[:, ins_idx]
            counts = np.bincount(ins_labels)
            label = np.argmax(counts)
            vote_label.append(label)
        acc = np.mean(vote_label == Yt)
        print(acc)

        print("=============Accuracy best=============")
        F = np.dot(self.K, best)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Yt_pseu = Y_pseudo[self.ns:].tolist()
        acc = np.mean(Yt_pseu == Yt)
        print(acc)

    def initialize_with_class(self, classifier):
        classifier.fit(self.Xs, self.Ys)
        Yt_pseu = classifier.predict(self.Xt)
        mu = 0.5
        N = 0
        for c in range(1, self.C + 1):
            e = np.zeros((self.ns + self.nt, 1))
            tt = self.Ys == c
            e[np.where(tt == True)] = 1.0 / len(self.Ys[np.where(self.Ys == c)])
            yy = Yt_pseu == c
            ind = np.where(yy == True)
            inds = [item + self.ns for item in ind]
            if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
                e[tuple(inds)] = 0.0
            else:
                e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
            e[np.isinf(e)] = 0
            N += np.dot(e, e.T)
        M = (1 - mu) * self.M0 + mu * N
        M /= np.linalg.norm(M, 'fro')
        left = np.dot(self.A + self.lamb * M + self.rho * self.L, self.K) \
               + self.eta * np.eye(self.ns + self.nt, self.ns + self.nt)
        Beta = np.dot(np.linalg.inv(left), np.dot(self.A, self.YY))
        return Beta

    def get_non_dominated(self, archive, archive_smr, archive_mmd):
        indices = []
        for index in range(len(archive)):
            if len(indices) == 0:
                indices.append(index)
            else:
                # go through each item to check
                be_dominated = False
                cur_smr = archive_smr[index]
                cur_mmd = archive_mmd[index]
                to_remove = []
                for store_index in indices:
                    if archive_smr[store_index] <= cur_smr and archive_mmd[store_index] <= cur_mmd:
                        be_dominated = True
                        break
                    elif archive_smr[store_index] > cur_smr and archive_mmd[store_index] > cur_mmd:
                        to_remove.append(store_index)

                if not be_dominated:
                    indices.append(index)
                    indices = [index for index in indices if index not in to_remove]
        return indices

    def re_initialize(self, pop, best, pos_min, pos_max, strategy=0):
        NBIT = best.shape[0] * best.shape[1]
        rand_pos = np.random.uniform(pos_min, pos_max, NBIT)
        rand_pos = np.reshape(rand_pos, (best.shape[0], best.shape[1]))

        if strategy == 0:
            # re-initialize randomly
            return rand_pos
        elif strategy == 1:
            # re-initinalize using best and the direction from
            # a population member to best
            ins_pos = pop[np.random.randint(0, len(pop) - 1)]
            rand_pos = rand_pos + 0.5 * (best - ins_pos)
            return rand_pos
        else:
            print("Unsupported re-initialization strategy.")
            sys.exit(1)

    def fit_predict(self, Beta):
        F = np.dot(self.K, Beta)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Yt_pseu = Y_pseudo[self.ns:]

        mu = 0.5
        N = 0
        for c in range(1, self.C + 1):
            e = np.zeros((self.ns + self.nt, 1))
            tt = self.Ys == c
            e[np.where(tt == True)] = 1.0 / len(self.Ys[np.where(self.Ys == c)])
            yy = Yt_pseu == c
            ind = np.where(yy == True)
            inds = [item + self.ns for item in ind]
            if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
                e[tuple(inds)] = 0.0
            else:
                e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
            e[np.isinf(e)] = 0
            N += np.dot(e, e.T)
        M = (1 - mu) * self.M0 + mu * N
        M /= np.linalg.norm(M, 'fro')
        left = np.dot(self.A + self.lamb * M + self.rho * self.L, self.K) \
               + self.eta * np.eye(self.ns + self.nt, self.ns + self.nt)
        Beta = np.dot(np.linalg.inv(left), np.dot(self.A, self.YY))

        # Now given the new beta, calculate the fitness
        SRM = np.linalg.norm(np.dot(self.YY.T - np.dot(Beta.T, self.K), self.A)) \
              + self.eta * np.linalg.multi_dot([Beta.T, self.K, Beta]).trace()
        MMD = self.lamb * np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([self.K, M, self.K]), Beta]).trace()
        fitness = SRM + MMD

        # Calcuate the accuracy
        F = np.dot(self.K, Beta)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Ys_pseu = Y_pseudo[:self.ns]
        acc_s = np.mean(Ys_pseu == self.Ys)
        Yt_pseu = Y_pseudo[self.ns:]
        acc_t = np.mean(Yt_pseu == self.Yt)

        return Beta, fitness, MMD, SRM, acc_s, acc_t


if __name__ == '__main__':
    source = np.genfromtxt("data/Source", delimiter=",")
    m = source.shape[1] - 1
    Xs = source[:, 0:m]
    Ys = np.ravel(source[:, m:m + 1])
    Ys = np.array([int(label) for label in Ys])

    target = np.genfromtxt("data/Target", delimiter=",")
    Xt = target[:, 0:m]
    Yt = np.ravel(target[:, m:m + 1])
    Yt = np.array([int(label) for label in Yt])

    r_meda = Random_MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10,
                         init_op=2, re_init_op=1, seed=1617)
    r_meda.evolve(Xs, Ys, Xt, Yt)
