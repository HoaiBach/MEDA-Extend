from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import Helpers
import Helpers as Pre
import GFK
import Paras


class MultiTransferProblem(IntegerProblem):

    def __init__(self, normalize, gfk_dim):
        """
        :param normalize: whether to normalize the data or not
        :param gfk_dim: what is the dimension of gfk
        """
        # eta can be tuned later through arguments
        self.eta = 0.1

        # first load the data
        source = np.genfromtxt("Source", delimiter=",")
        self.no_features, self.no_src_instances = source.shape[1] - 1, source.shape[0]
        self.Xs = source[:, 0:self.no_features]
        self.Ys = np.ravel(source[:, self.no_features:self.no_features + 1])
        self.Ys = np.array([int(label) for label in self.Ys])

        target = np.genfromtxt("Target", delimiter=",")
        self.no_tar_instances = target.shape[0]
        self.Xt = target[:, 0: self.no_features]
        self.Yt = np.ravel(target[:, self.no_features:self.no_features + 1])
        self.Yt = np.array([int(label) for label in self.Yt])

        if normalize:
            self.Xs, self.Xt = Pre.normalize_data(self.Xs, self.Xt)

        # make sure the class indices start from 1
        self.no_class = len(np.unique(self.Ys))
        if self.no_class > np.max(self.Ys):
            self.Ys = self.Ys + 1
            self.Yt = self.Yt + 1

        # transform data using gfk
        gfk = GFK.GFK(dim=gfk_dim)
        _, Xs_new, Xt_new = gfk.fit(self.Xs, self.Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        X /= np.linalg.norm(X, axis=0)
        self.X = X.T
        self.Xs = X[:, :self.no_src_instances].T
        self.Xt = X[:, self.no_src_instances:].T

        self.YY = np.zeros((self.no_src_instances, self.no_class))
        for c in range(1, self.no_class + 1):
            ind = np.where(self.Ys == c)
            self.YY[ind, c - 1] = 1
        self.YY = np.vstack((self.YY, np.zeros((self.no_tar_instances, self.no_class))))

        # build some matrices that are not changed in the evaluation
        self.K = self.kernel('rbf', self.X.T, X2=None, gamma=0.5)
        self.A = np.diagflat(np.vstack((np.ones((self.no_src_instances, 1)),
                                        np.zeros((self.no_tar_instances, 1)))))
        e = np.vstack((1.0 / self.no_src_instances * np.ones((self.no_src_instances, 1)),
                       -1.0 / self.no_tar_instances * np.ones((self.no_tar_instances, 1))))
        self.M0 = e * e.T * self.no_class

        self.L = Pre.laplacian_matrix(self.X, k=10)

        super(MultiTransferProblem, self).__init__()

        # initialize the objectives for multi-objective optimisation
        self.number_of_variables = self.no_tar_instances
        self.number_of_constraints = 0

        self.obj_labels = ['Discrepancy', 'Manifold']
        self.number_of_objectives = len(self.obj_labels)
        self.obj_directions = [self.MINIMIZE, ]*self.number_of_objectives

        # the lower bound and upper bound of the decision variables
        # are integers from 1 to the number of classes
        self.lower_bound = [1 for _ in range(self.number_of_variables)]
        self.upper_bound = [self.no_class for _ in range(self.number_of_variables)]

        IntegerSolution.lower_bound = self.lower_bound
        IntegerSolution.upper_bound = self.upper_bound

        # now set up the initialized label vector
        self.init_label_vector = self.initialize_label_vector(mode='full')
        self.init_index = 0

    def initialize_label_vector(self, mode='full'):
        """
        Build a set of label vectors, which will be used for initialising the
        population
        :param mode: should we use full opposite mechanism or just random
        :return:
        """
        label_vector = np.random.randint(low=1, high=self.no_class + 1,
                                                   size=(Paras.N_IND, self.no_tar_instances))

        # now use the opposite mechanism to do the initialization
        if mode == 'full':
            Helpers.opposite_init(label_vector, min_pos=1, max_pos=self.no_class)

        # the first vector is initialised by the 1NN
        classifier = KNeighborsClassifier(1)
        classifier.fit(self.Xs, self.Ys)
        Yt_pseu = classifier.predict(self.Xt)
        for label_index in range(len(label_vector[0])):
            label_vector[0][label_index] = Yt_pseu[label_index]

        # the second vector is initialised by the manifold
        left = np.dot(self.A + self.L, self.K) \
               + self.eta * np.eye(self.no_src_instances + self.no_tar_instances,
                                   self.no_src_instances + self.no_tar_instances)
        beta = np.dot(np.linalg.inv(left), np.dot(self.A, self.YY))

        F = np.dot(self.K, beta)
        Y_pseu = np.argmax(F, axis=1) + 1
        Yt_pseu = Y_pseu[self.no_src_instances:].tolist()
        for label_index in range(len(label_vector[1])):
            label_vector[1][label_index] = Yt_pseu[label_index]

        return label_vector

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        """
        Method to create a solution for JmetalPy based on the initialised vector
        """

        if self.init_index == len(self.init_label_vector):
            self.init_index = 0
        pseu_label = self.init_label_vector[self.init_index]
        new_solution.variables = []
        for _, value in enumerate(pseu_label):
            new_solution.variables.append(value)
        self.init_index = self.init_index+1

        return new_solution

    def cross_domain_error(self, tar_label):
        """
        This method trains a classification algorithm using source label, to predict target instances
        Then, it trains a classification algorithm using target label, to predict source instances
        Calculate the average classification error rates
        :param tar_label: labels of target instances
        :return: average error from both domains
        """
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(self.Xs, self.Ys)
        acc_1 = (np.sum(clf.predict(self.Xt) == tar_label) + 0.0)/len(tar_label)
        clf.fit(self.Xt, tar_label)
        acc_2 = (np.sum(clf.predict(self.Xs) == self.Ys) + 0.0)/len(self.Ys)
        return 1.0-(acc_1+acc_2)/2

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        """
        Evaluate the fitness values of the given solution
        :param solution:
        :return:
        """
        Yt_pseu = np.array(solution.variables)
        Y = np.concatenate((self.Ys, Yt_pseu))
        F = np.zeros((self.no_src_instances + self.no_tar_instances, self.no_class))
        for index, l in enumerate(Y):
            F[index][l - 1] = 1.0

        # calculate the manifold objective
        manifold = np.linalg.multi_dot([F.T, self.L, F]).trace()

        # calculate the discrepancy objective
        M = self.mmd_matrix(Yt_pseu)
        discrepancy = np.linalg.multi_dot([F.T, M, F]).trace()

        # calculate the cross domain error
        # error = self.cross_domain_error(Yt_pseu)

        solution.objectives[0] = discrepancy
        solution.objectives[1] = manifold
        # solution.objectives[2] = manifold # error

        return solution

    def mmd_matrix(self, target_label):
        '''
        Calaculate the MMD matrix based on the label of the target instances
        :param target_label:
        :return: MMD matrix
        '''
        mu = 0.5
        N = 0
        for c in range(1, self.no_class + 1):
            e = np.zeros((self.no_src_instances + self.no_tar_instances, 1))
            tt = self.Ys == c
            e[np.where(tt == True)] = 1.0 / len(self.Ys[np.where(self.Ys == c)])
            yy = target_label == c
            ind = np.where(yy == True)
            inds = [item + self.no_src_instances for item in ind]
            if len(target_label[np.where(target_label == c)]) == 0:
                e[tuple(inds)] = 0.0
            else:
                e[tuple(inds)] = -1.0 / len(target_label[np.where(target_label == c)])
            e[np.isinf(e)] = 0
            N += np.dot(e, e.T)
        M = (1 - mu) * self.M0 + mu * N
        M /= np.linalg.norm(M, 'fro')
        return M

    def step_discrepancy(self, solutions):
        '''
        Perform local search using gradient on the discrepancy objective
        The step_discrepancy applies the local search to all solutions
        :param solutions: contains solutions for local search
        :return:
        '''
        grad_solutions = []
        for sol in solutions:
            Yt_pseu = np.array(sol.variables)
            M = self.mmd_matrix(Yt_pseu)
            left = np.dot(self.A + M, self.K) \
                   + self.eta * np.eye(self.no_src_instances + self.no_tar_instances,
                                       self.no_src_instances + self.no_tar_instances)
            beta = np.dot(np.linalg.inv(left), np.dot(self.A, self.YY))

            F = np.dot(self.K, beta)
            Y_pseu = np.argmax(F, axis=1) + 1
            Yt_pseu = Y_pseu[self.no_src_instances:].tolist()

            new_solution = IntegerSolution(
                self.lower_bound,
                self.upper_bound,
                self.number_of_objectives,
                self.number_of_constraints)

            new_solution.variables = []
            for _, value in enumerate(Yt_pseu):
                new_solution.variables.append(value)

            grad_solutions.append(new_solution)

        return grad_solutions

    def kernel(self, ker, X, X2, gamma):
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

    def get_name(self) -> str:
        return "Multi-objective Transfer learning"
