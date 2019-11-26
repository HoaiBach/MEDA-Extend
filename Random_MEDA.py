# encoding=utf-8
"""
    Created on 10:40 2018/12/14
    @author: Bach Nguyen
"""

import numpy as np
from sklearn import metrics, neighbors
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
import time
import Helpers as Pre
import random
import os


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
    if nb_source == 0 or nb_target == 0:
        return -2
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
                 init_op=0, re_init_op=0, archive_size=2, random_rate=0.5, T=10, run=1):
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
        self.re_init_op = re_init_op  # 0-random, 1-using best, 2-using existing archive

        self.run = run
        seed = 1617 * run
        np.random.seed(seed)
        self.archive_size = archive_size
        self.random_rate = random_rate
        self.T = 10

    def evolve(self, Xs, Ys, Xt, Yt):
        self.Xs = Xs
        self.Ys = Ys
        self.Xt = Xt
        self.Yt = Yt

        self.ns, self.nt = Xs.shape[0], Xt.shape[0]
        self.C = len(np.unique(Ys))

        f_out = open(str(self.run) + '.txt', 'w')
        toPrint = ("Random_rate: %f, archive size: %d\n" % (self.random_rate, self.archive_size))

        start = time.time()
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
        self.L = laplacian_matrix(X.T, self.p)

        self.YY = np.zeros((self.ns, self.C))
        for c in range(1, self.C + 1):
            ind = np.where(self.Ys == c)
            self.YY[ind, c - 1] = 1
        self.YY = np.vstack((self.YY, np.zeros((self.nt, self.C))))

        N = 10
        GEN = self.T
        pos_min = -10
        pos_max = 10
        pop = []
        pop_mmd = [sys.float_info.max] * N
        pop_srm = [sys.float_info.max] * N
        pop_fit = [sys.float_info.max] * N
        pop_src_acc = [sys.float_info.max] * N
        pop_tar_acc = [sys.float_info.max] * N
        pop_label = [[1]] * N
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
                classifier = KNeighborsClassifier(2 * i + 1)
                beta = self.initialize_with_classifier(classifier)
                pop.append(beta)
        elif self.init_op == 2:
            # using differnet classifiers
            classifiers = list([])
            classifiers.append(KNeighborsClassifier(1))
            classifiers.append(KNeighborsClassifier(3))
            classifiers.append(KNeighborsClassifier(5))
            classifiers.append(SVC(kernel="linear", C=0.025, random_state=np.random.randint(2 ** 10)))
            classifiers.append(SVC(kernel="rbf", C=1, gamma=2, random_state=np.random.randint(2 ** 10)))
            classifiers.append(GaussianProcessClassifier(1.0 * RBF(1.0), random_state=np.random.randint(2 ** 10)))
            classifiers.append(GaussianNB())
            classifiers.append(DecisionTreeClassifier(max_depth=5, random_state=np.random.randint(2 ** 10)))
            classifiers.append(
                RandomForestClassifier(max_depth=5, n_estimators=10, random_state=np.random.randint(2 ** 10)))
            classifiers.append(AdaBoostClassifier(random_state=np.random.randint(2 ** 10)))
            assert len(classifiers) == N
            for i in range(N):
                beta = self.initialize_with_classifier(classifiers[i])
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
        archive_label = []
        best = None
        best_fitness = sys.float_info.max
        best_src_acc = -sys.float_info.max
        best_tar_acc = -sys.float_info.max
        best_mmd = 0
        best_srm = 0

        toPrint += ("# Form index: fitness, source accuracy, target accuracy\n")
        for g in range(GEN):
            toPrint += ('==============Gen %d===============\n' % g)
            for index, ind in enumerate(pop):

                # refine the position using gradient descent
                new_position, new_fitness, new_mmd, new_srm, new_src_acc, new_tar_acc, new_label = self.fit_predict(
                    pop[index])
                toPrint += ("%d: %f, %f, %f\n" % (index, new_fitness, new_src_acc, new_tar_acc))

                # create new position based on crossover and mutation

                # if the fitness is not improved, store the previous position in the archive
                # randomly create a new position based on the best
                # store the current position to archive
                if pop_fit[index] <= new_fitness:
                    new_position = self.re_initialize(pop, best, pos_min, pos_max, archive_label,
                                                      strategy=self.re_init_op)
                    new_position, new_fitness, new_mmd, new_srm, new_src_acc, new_tar_acc, new_label = self.fit_predict(
                        new_position)
                    toPrint += ("Reset %d: %f, %f, %f\n" % (index, new_fitness, new_src_acc, new_tar_acc))

                    # append the old ind
                    archive.append(pop[index])
                    archive_fit.append(pop_fit[index])
                    archive_mmd.append(pop_mmd[index])
                    archive_srm.append(pop_srm[index])
                    archive_src_acc.append(pop_src_acc[index])
                    archive_tar_acc.append(pop_tar_acc[index])
                    archive_label.append(pop_label[index])

                # now update the new position with its fitness
                pop[index] = new_position
                pop_fit[index] = new_fitness
                pop_mmd[index] = new_mmd
                pop_srm[index] = new_srm
                pop_src_acc[index] = new_src_acc
                pop_tar_acc[index] = new_tar_acc
                pop_label[index] = new_label

                # update best if necessary
                if new_fitness < best_fitness:
                    best = np.copy(pop[index])
                    best_fitness = new_fitness
                    best_src_acc = new_src_acc
                    best_tar_acc = new_tar_acc
                    best_srm = new_srm
                    best_mmd = new_mmd

            toPrint += ("Best: %f, %f, %f\n" % (best_fitness, best_src_acc, best_tar_acc))

        archive.append(best)
        archive_fit.append(best_fitness)
        archive_mmd.append(best_mmd)
        archive_srm.append(best_srm)
        archive_src_acc.append(best_src_acc)
        archive_tar_acc.append(best_tar_acc)

        time_eslape = (time.time() - start)
        nd_indices = self.get_non_dominated(archive, archive_srm, archive_mmd)

        all_labels = []
        toPrint += ("========From all archive========\n")
        all_indices = range(len(archive))
        class_indices = all_indices
        for index in class_indices:
            ind = archive[index]
            toPrint += ("%d: %f, %f, %f, %f, %f\n"
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
        label_to_return = vote_label
        acc_to_return = acc
        toPrint += ("Accuracy archive:" + str(acc) + "\n")

        all_labels = []
        toPrint += ("========From non-dominated========\n")
        class_indices = nd_indices
        for index in class_indices:
            ind = archive[index]
            toPrint += ("%d: %f, %f, %f, %f, %f\n"
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
        toPrint += ("Accuracy non-dominated:" + str(acc) + "\n")

        toPrint += ("===================================\n")
        F = np.dot(self.K, best)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Yt_pseu = Y_pseudo[self.ns:].tolist()
        acc = np.mean(Yt_pseu == Yt)
        toPrint += ("Accuracy best:" + str(acc) + "\n")
        toPrint += ("Execution time: " + str(time_eslape) + "\n")

        f_out.write(toPrint)
        f_out.close()

        return label_to_return, acc_to_return

    def initialize_with_label(self, label):
        Yt_pseu = label
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

    def initialize_with_classifier(self, classifier):
        classifier.fit(self.Xs, self.Ys)
        Yt_pseu = classifier.predict(self.Xt)
        return self.initialize_with_label(Yt_pseu)

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

    def re_initialize(self, pop, best, pos_min, pos_max, archive_label, strategy=0):
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
        elif strategy == 2:
            # re-initialize using the archive set
            if len(archive_label) < self.archive_size:
                return rand_pos
            else:
                pseu_labels = []
                for ins_index in range(self.nt):
                    list_labels = [archive_label[i][ins_index] for i in range(len(archive_label))]
                    ins_label = np.random.choice(np.unique(list_labels))
                    pseu_labels.append(ins_label)
                return self.initialize_with_label(np.array(pseu_labels))
        elif strategy == 3:
            # re-initialize using the archive set
            if len(archive_label) < self.archive_size or np.random.rand() <= self.random_rate:
                return rand_pos
            else:
                pseu_labels = []
                for ins_index in range(self.nt):
                    list_labels = [archive_label[i][ins_index] for i in range(len(archive_label))]
                    ins_label = np.random.choice(np.unique(list_labels))
                    pseu_labels.append(ins_label)
                return self.initialize_with_label(np.array(pseu_labels))
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
        MMD = np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([self.K, self.lamb * M + self.rho * self.L, self.K]), Beta]).trace()
        fitness = SRM + MMD

        # Calcuate the accuracy
        F = np.dot(self.K, Beta)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Ys_pseu = Y_pseudo[:self.ns]
        acc_s = np.mean(Ys_pseu == self.Ys)
        Yt_pseu = Y_pseudo[self.ns:]
        acc_t = np.mean(Yt_pseu == self.Yt)

        return Beta, fitness, MMD, SRM, acc_s, acc_t, Yt_pseu


def laplacian_matrix(data, k):
    """
    :param data: containing data points,
    :param k: the number of neighbors considered (this distance metric is cosine,
    and the weights are measured by cosine)
    :return:
    """
    nn = neighbors.NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nn.fit(data)
    dist, nn = nn.kneighbors(return_distance=True)
    sim = np.zeros((len(data), len(data)))
    for ins_index in range(len(sim)):
        dist_row = dist[ins_index]
        nn_row = nn[ins_index]
        for dist_value, ind_index in zip(dist_row, nn_row):
            sim[ins_index][ind_index] = 1.0 - dist_value
            sim[ind_index][ins_index] = 1.0 - dist_value
    for i in range(len(sim)):
        sim[i][i] = 1.0

    S = [np.sum(row) for row in sim]

    for i in range(len(sim)):
        sim[i] = [sim[i][j]/(S[i]*S[j])**0.5 for j in range(len(sim))]

    L = np.identity(len(sim)) - sim
    return L


if __name__ == '__main__':
    run = int(sys.argv[1])
    random_seed = 1617 * run
    normalize = int(sys.argv[2]) == 1
    dim = int(sys.argv[3])
    init_op = 2  # 0-random, 1- KNN (diff K), 2-10 diff classifiers
    re_init_op = 3  # 0-random, 1-using best, 2-label, 3- mixed random label
    archive_size = 10  # size of archive to be ok for using in creating new (using with label/mix label and random)
    random_rate = 0.5  # arg[4] can be 1,2,3,.., 10 -> 0.1, 0.2, 0.3,...,1.0

    source = np.genfromtxt("Source", delimiter=",")
    m = source.shape[1] - 1
    Xs = source[:, 0:m]
    Ys = np.ravel(source[:, m:m + 1])
    Ys = np.array([int(label) for label in Ys])

    target = np.genfromtxt("Target", delimiter=",")
    Xt = target[:, 0:m]
    Yt = np.ravel(target[:, m:m + 1])
    Yt = np.array([int(label) for label in Yt])

    if normalize:
        Xs, Xt = Pre.normalize_data(Xs, Xt)
    C = len(np.unique(Ys))
    if C > np.max(Ys):
        Ys = Ys + 1
        Yt = Yt + 1
    np.random.seed(random_seed)
    random.seed(random_seed)

    r_meda = Random_MEDA(kernel_type='rbf', dim=dim, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10,
                         init_op=init_op, re_init_op=re_init_op, run=run, archive_size=10)
    r_meda.evolve(Xs, Ys, Xt, Yt)
