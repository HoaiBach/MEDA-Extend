# encoding=utf-8
"""
    Created on 10:40 2018/12/14
    @author: Bach Nguyen
"""

import numpy as np
from sklearn import metrics
from sklearn import svm

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
    def __init__(self, kernel_type='primal', dim=30, lamb=1, rho=1.0, eta=0.1, p=10, gamma=1.0, T=10, seed=1617):
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
        self.YY = None # 1 hot coding
        self.L = 0
        # self.Yt_pseu = None

        np.random.seed(seed)

    def evolve(self, Xs, Ys, Xt, Yt):
        self.Xs = Xs
        self.Ys = Ys
        self.Xt = Xt
        self.Yt = Yt

        self.ns, self.nt = Xs.shape[0], Xt.shape[0]
        self.C = len(np.unique(Ys))

        # Transform data using gfk
        gfk = GFK.GFK(dim=self.dim)
        _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        X /= np.linalg.norm(X, axis=0)

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
        fits = []
        NBIT = (self.ns+self.nt)*self.C

        # initialization
        for i in range(N):
            poistion = np.random.uniform(pos_min, pos_max, NBIT)
            beta = np.reshape(poistion, (self.ns+self.nt, self.C))
            pop.append(beta)
            fits.append(sys.float_info.max)

        # evolution
        archive = []
        best = None
        best_fitness = sys.float_info.max
        best_acc_s = -sys.float_info.max
        best_acc_t = -sys.float_info.max
        for g in range(GEN):
            print('==============Gen %d===============' % g)
            for index, ind in enumerate(pop):
                new_position, fitness, ind_acc_s, ind_acc_t = self.fit_predict(pop[index])
                print("Ind %d has fitness of %f and source accuracy %f and target accuracy %f." % (index, fitness, ind_acc_s, ind_acc_t))

                # if the fitness is not improved, store the previous position in the archive
                # randomly create a new position based on the best
                # store the current position to archive
                if fits[index] <= fitness:
                    print("***Reset ind %d" % index)
                    new_position = self.re_initialize(pop, best, pos_min, pos_max)
                    new_position, fitness, ind_acc_s, ind_acc_t = self.fit_predict(new_position)
                    archive.append(pop[index])
                    print("archive size %d***" %len(archive))

                pop[index] = new_position
                fits[index] = fitness

                # update best if necessary
                if fitness < best_fitness:
                    best = np.copy(pop[index])
                    best_fitness = fitness
                    best_acc_s = ind_acc_s
                    best_acc_t = ind_acc_t

            print("Best fitness of %f and source accuracy %f and target accuracy %f." % (best_fitness, best_acc_s, best_acc_t))

        archive.append(best)
        print("Final archive size %d" % len(archive))
        all_labels = []
        for ind in pop:
            Beta = np.copy(ind)
            Beta = np.reshape(Beta, (self.ns+self.nt, self.C))
            F = np.dot(self.K, Beta)
            Y_pseudo = np.argmax(F, axis=1) + 1
            Yt_pseu = Y_pseudo[self.ns:].tolist()
            all_labels.append(Yt_pseu)
        all_labels = np.array(all_labels)
        vote_label = []
        for ins_idx in range(all_labels.shape[1]):
            ins_labels = all_labels[:,ins_idx]
            counts = np.bincount(ins_labels)
            label = np.argmax(counts)
            vote_label.append(label)
        acc = np.mean(vote_label == Yt)
        print(acc)

    def re_initialize(self, pop, best, pos_min, pos_max):
        NBIT = best.shape[0] * best.shape[1]
        rand_pos = np.random.uniform(pos_min, pos_max, NBIT)
        rand_pos = np.reshape(rand_pos, (best.shape[0], best.shape[1]))
        ins_pos = pop[np.random.randint(0, len(pop)-1)]
        rand_pos = rand_pos+0.5*(best-ins_pos)
        return rand_pos

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
              + self.eta*np.linalg.multi_dot([Beta.T, self.K, Beta]).trace()
        MMD = self.lamb*np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([self.K, M, self.K]), Beta]).trace()
        fitness = SRM + MMD

        # Calcuate the accuracy
        F = np.dot(self.K, Beta)
        Y_pseudo = np.argmax(F, axis=1) + 1
        Ys_pseu = Y_pseudo[:self.ns]
        acc_s = np.mean(Ys_pseu == self.Ys)
        Yt_pseu = Y_pseudo[self.ns:]
        acc_t = np.mean(Yt_pseu == self.Yt)

        return Beta, fitness, acc_s, acc_t


if __name__ == '__main__':
    import sys
    run = int(sys.argv[1])
    random_seed = 1617*run

    source = np.genfromtxt("data/Source", delimiter=",")
    m = source.shape[1] - 1
    Xs = source[:, 0:m]
    Ys = np.ravel(source[:, m:m + 1])
    Ys = np.array([int(label) for label in Ys])

    target = np.genfromtxt("data/Target", delimiter=",")
    Xt = target[:, 0:m]
    Yt = np.ravel(target[:, m:m + 1])
    Yt = np.array([int(label) for label in Yt])

    r_meda = Random_MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10, seed=random_seed)
    r_meda.evolve(Xs, Ys, Xt, Yt)
