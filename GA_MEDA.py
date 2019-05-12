from deap import base, creator, tools
import numpy as np
import GFK
from sklearn import svm, metrics
import random


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


class GA_MEDA:

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
        self.YY = None  # 1 hot coding
        self.L = 0

        np.random.seed(seed)
        random.seed(seed)

    def fit_predict(self, Yt_pseu):
        """
        Calculate the fitness function of a label array of target instances
        :param Yt_pseu: the pseudolabel of target instances, which are in a chromosome.
        :return:
        """
        # Based on the Yt_pseu, calculate the optimal beta.
        Yt_pseu = np.array(Yt_pseu)
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
        # Y_pseudo = np.argmax(F, axis=1) + 1
        # Ys_pseu = Y_pseudo[:self.ns]
        # acc_s = np.mean(Ys_pseu == self.Ys)
        # Yt_pseu = Y_pseudo[self.ns:]
        # acc_t = np.mean(Yt_pseu == self.Yt)

        return fitness,

    def evolve(self, Xs, Ys, Xt, Yt):
        """
        Running GA algorithms, where each individual is a set of target pseudo labels.
        :return: the best solution of GAs.
        """
        # initialize the data
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

        # parameters for GA
        self.N_BIT = self.nt
        self.N_GEN = 100
        self.N_IND = 30
        self.MUTATION_RATE = 1.0 / self.N_BIT
        self.CXPB = 0.2
        self.MPB = 0.8

        self.pos_min = 1
        self.pos_max = self.C

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Ind", list, fitness=creator.FitnessMin)
        creator.create("Pop", list)

        # for creating the population
        toolbox = base.Toolbox()
        toolbox.register("bit", random.randint, a=self.pos_min, b=self.pos_max)
        toolbox.register("ind", tools.initRepeat, creator.Ind, toolbox.bit, n=self.N_BIT)
        toolbox.register("pop", tools.initRepeat, creator.Pop, toolbox.ind, n=self.N_IND)
        # for evaluation
        toolbox.register("evaluate", self.fit_predict)
        # for genetic operators
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("crossover", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=self.pos_min, up=self.pos_max,
                         indpb=self.MUTATION_RATE)

        # evolutionary process
        pop = toolbox.pop()

        start_fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, start_fitnesses):
            ind.fitness.values = fit

        hof = tools.HallOfFame(maxsize=1)
        hof.update(pop)

        for g in range(self.N_GEN):
            print("=========== Iteration %d ===========" %g)

            # selection
            offspring = toolbox.select(pop, len(pop))
            offspring = map(toolbox.clone, offspring)

            # applying crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < self.CXPB:
                    toolbox.crossover(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # applying mutation
            for mutant in offspring:
                if np.random.rand() < self.MPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = tools.selBest(offspring+list(hof), len(pop))
            hof.update(pop)
            best_ind = tools.selBest(pop, 1)[0]
            print("Best fitness %f " %best_ind.fitness.values)

        best_ind = tools.selBest(pop,1)[0]
        print("=========== Final result============")
        print(best_ind)
        acc = np.mean(best_ind == self.Yt)
        print("Accuracy: %f" %acc)


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

    g_meda = GA_MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10, seed=random_seed)
    g_meda.evolve(Xs, Ys, Xt, Yt)