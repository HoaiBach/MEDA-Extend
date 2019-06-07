import random

import Helpers as Pre
import numpy as np
from deap import base, creator, tools
from sklearn import metrics, neighbors
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import Helpers
import GFK
import MEDA


# from scoop import futures
# toolbox.register("map", futures.map)
# for parallel
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
    if len(np.unique((target_X))) < 2:
        return -2
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


kernel_type = 'rbf'
dim = 20
lamb = 10
rho = 1.0
eta = 0.1
p = 10
gamma = 0.5
T = 10

Xs = 0
Ys = 0
Xt = 0
Yt = 0
YY = 0

C = 0
ns = 0
nt = 0

M0 = 0
K = 0
A = 0
e = 0

toolbox = base.Toolbox()

def estimate_mu(_X1, _Y1, _X2, _Y2):
    return 0.5
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


def fitness_evaluation(Yt_pseu):
    '''
    Calculate the fitness of Yt_pseu, it is expensive to calculate beta
    so from Yt_pseu, set beta as the one-hot code
    :param Yt_pseu: size nt*c
    :return: fitness of Yt_pseu
    '''
    Yt_pseu = np.array(Yt_pseu)
    Y = np.concatenate((Ys, Yt_pseu))
    F = np.zeros((ns+nt, C))
    for index, l in enumerate(Y):
        F[index][l-1] = 1
    mu = estimate_mu(Xs, Ys, Xt, Yt_pseu)
    # have to update the matrix M
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((ns + nt, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
            e[tuple(inds)] = 0.0
        else:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        e[np.isinf(e)] = 0
        N += np.dot(e, e.T)
    M = (1 - mu) * M0 + mu * N

    fitness = np.linalg.multi_dot([F.T, lamb * M + rho * L, F]).trace()

    return fitness


def beta_predict(beta):
    """
    predict the label given the beta matrix
    :param: beta
    :return: predict label
    """
    N = 0
    # re-calculate the soft label again based on the obtained Beta
    F = np.dot(K, beta)
    Y_pseudo = np.argmax(F, axis=1) + 1
    Yt_pseu = Y_pseudo[ns:]
    mu = estimate_mu(Xs, Ys, Xt, Yt_pseu)
    # have to update the matrix M
    for c in range(1, C + 1):
        e = np.zeros((ns + nt, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
            e[tuple(inds)] = 0.0
        else:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        e[np.isinf(e)] = 0
        N += np.dot(e, e.T)
    M = (1 - mu) * M0 + mu * N

    M /= np.linalg.norm(M, 'fro')
    left = np.dot(A + lamb * M + rho * L, K) \
           + eta * np.eye(ns + nt, ns + nt)
    beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

    F = np.dot(K, beta)
    Y_pseu = np.argmax(F, axis=1) + 1
    Yt_pseu = Y_pseu[ns:].tolist()

    return Yt_pseu

def label_evolve(ind):
    '''
    Given an individual (pseudo label for target isntances)
    evolving new psudo-labels by a single step meda, then return it
    :param ind: an ind with Yt_psuedo
    :return: new pseudo
    '''
    Yt_pseu = np.array([ind[index] for index in range(len(ind))])
    mu = estimate_mu(Xs, Ys, Xt, Yt_pseu)
    N = 0
    # have to update the matrix M
    for c in range(1, C + 1):
        e = np.zeros((ns + nt, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
            e[tuple(inds)] = 0.0
        else:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        e[np.isinf(e)] = 0
        N += np.dot(e, e.T)
    M = (1 - mu) * M0 + mu * N

    M /= np.linalg.norm(M, 'fro')
    left = np.dot(A + lamb * M + rho * L, K) \
           + eta * np.eye(ns + nt, ns + nt)
    beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

    F = np.dot(K, beta)
    Y_pseu = np.argmax(F, axis=1) + 1
    Yt_pseu = Y_pseu[ns:].tolist()
    return Yt_pseu


def label_phase(ind):
    '''
    Calculate the fitness of ind, ind has the label part and beta part
    :param ind: individual to calculate the fitness
    :return: the fitness
    '''
    Yt_pseu = np.array([ind[index] for index in range(len(ind))])
    beta = ind.beta
    mu = estimate_mu(Xs, Ys, Xt, Yt_pseu)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((ns + nt, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
            e[tuple(inds)] = 0.0
        else:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        e[np.isinf(e)] = 0
        N += np.dot(e, e.T)
    M = (1 - mu) * M0 + mu * N
    M /= np.linalg.norm(M, 'fro')
    F = np.dot(K, beta)

    # Now given the new beta, calculate the fitness
    SRM = np.linalg.norm(np.dot(YY.T - F.T, A)) \
          + eta * np.linalg.multi_dot([beta.T, K, beta]).trace()
    MMD = np.linalg.multi_dot([beta.T,
                               np.linalg.multi_dot([K, lamb * M + rho * L, K]),
                               beta]).trace()
    fitness = SRM + MMD

    return fitness


def beta_phase(ind):
    '''
    Based on the label in ind, calculate the new Beta
    :param ind: individual whose beta needs to be updated
    :return: the beta matrix based on the ind
    '''
    Yt_pseu = np.array([ind[index] for index in range(len(ind))])
    mu = estimate_mu(Xs, Ys, Xt, Yt_pseu)
    N = 0
    for c in range(1, C + 1):
        e = np.zeros((ns + nt, 1))
        tt = Ys == c
        e[np.where(tt == True)] = 1.0 / len(Ys[np.where(Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) == 0:
            e[tuple(inds)] = 0.0
        else:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        e[np.isinf(e)] = 0
        N += np.dot(e, e.T)
    M = (1 - mu) * M0 + mu * N
    M /= np.linalg.norm(M, 'fro')
    left = np.dot(A + lamb * M + rho * L, K) \
           + eta * np.eye(ns + nt, ns + nt)
    beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

    return beta


def evolve(Xsource, Ysource, Xtarget, Ytarget, file, mutation_rate, full_init):
    """
    Running GA algorithms, where each individual is a set of target pseudo labels.
    :return: the best solution of GAs.
    """
    global ns, nt, C, Xs, Ys, Xt, Yt, YY, K, A, e, M0, L, archive
    archive = []
    Xs = Xsource
    Ys = Ysource
    Xt = Xtarget
    Yt = Ytarget
    ns, nt = Xs.shape[0], Xt.shape[0]
    C = len(np.unique(Ys))

    # Transform data using gfk
    gfk = GFK.GFK(dim=dim)
    _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
    Xs_new, Xt_new = Xs_new.T, Xt_new.T
    X = np.hstack((Xs_new, Xt_new))
    X /= np.linalg.norm(X, axis=0)
    Xs = X[:, :ns].T
    Xt = X[:, ns:].T

    # build some matrices that are not changed
    K = kernel(kernel_type, X, X2=None, gamma=gamma)
    A = np.diagflat(np.vstack((np.ones((ns, 1)), np.zeros((nt, 1)))))
    e = np.vstack((1.0 / ns * np.ones((ns, 1)), -1.0 / nt * np.ones((nt, 1))))
    M0 = e * e.T * C
    L = Helpers.laplacian_matrix(X.T, p)

    YY = np.zeros((ns, C))
    for c in range(1, C + 1):
        ind = np.where(Ys == c)
        YY[ind, c - 1] = 1
    YY = np.vstack((YY, np.zeros((nt, C))))

    pos_min = 1
    pos_max = C

    # parameters for GA
    N_BIT = nt
    N_GEN = 10
    N_IND = 100
    MUTATION_RATE = 1.0/N_BIT*C
    MPB = mutation_rate
    CXPB = 1-mutation_rate

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Ind", list, fitness=creator.FitnessMin)
    creator.create("Pop", list)

    # for creating the population
    toolbox.register("bit", random.randint, a=pos_min, b=pos_max)
    toolbox.register("ind", tools.initRepeat, creator.Ind, toolbox.bit, n=N_BIT)
    toolbox.register("pop", tools.initRepeat, creator.Pop, toolbox.ind)
    # for evaluation
    toolbox.register("evaluate", fitness_evaluation)
    # for genetic operators
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("crossover", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=pos_min, up=pos_max,
                     indpb=MUTATION_RATE)
    # pool = multiprocessing.Pool(4)
    # toolbox.register("map", pool.map)
    # initialize some individuals by predefined classifiers
    pop = toolbox.pop(n=N_IND)

    classifiers = list([])
    classifiers.append(KNeighborsClassifier(1))
    # classifiers.append(SVC(kernel="linear", C=0.025, random_state=np.random.randint(2 ** 10)))
    # classifiers.append(GaussianProcessClassifier(1.0 * RBF(1.0), random_state=np.random.randint(2 ** 10)))
    # classifiers.append(KNeighborsClassifier(3))
    # classifiers.append(SVC(kernel="rbf", C=1, gamma=2, random_state=np.random.randint(2 ** 10)))
    # classifiers.append(DecisionTreeClassifier(max_depth=5, random_state=np.random.randint(2 ** 10)))
    # classifiers.append(KNeighborsClassifier(5))
    # classifiers.append(GaussianNB())
    # classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10, random_state=np.random.randint(2 ** 10)))
    # classifiers.append(AdaBoostClassifier(random_state=np.random.randint(2 ** 10)))

    step = N_IND/len(classifiers)
    for ind_index, classifier in enumerate(classifiers):
        classifier.fit(Xs, Ys)
        Yt_pseu = classifier.predict(Xt)
        for bit_idex, value in enumerate(Yt_pseu):
            pop[ind_index*step][bit_idex] = value

    Helpers.opposite_init(pop, pos_min, pos_max, full_init)

    # evaluate the initialized populations
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit,

    hof = tools.HallOfFame(maxsize=1)
    hof.update(pop)

    for g in range(N_GEN):
        file.write("*****Iteration %d*****" % g)
        # selection
        offspring = toolbox.select(pop, len(pop))
        offspring = map(toolbox.clone, offspring)

        # applying crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < CXPB:
                toolbox.crossover(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # applying mutation
        for mutant in offspring:
            if np.random.rand() < MPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fitness in zip(invalid_ind, fitnesses):
                ind.fitness.values = fitness,

        # now select the best individual from offspring
        # pass it to the single step meda to refine the label
        best_ind = tools.selBest(offspring, 1)[0]
        Yt_pseu = label_evolve(best_ind)
        new_ind = toolbox.ind()
        for index, label in enumerate(Yt_pseu):
            new_ind[index] = label
        new_ind.fitness.values = fitness_evaluation(new_ind),
        offspring.append(new_ind)

        # The population is entirely replaced by the offspring
        pop[:] = tools.selBest(offspring + list(hof), len(pop))
        hof.update(pop)
        file.write('Average distance: %f' %(Helpers.pop_distance(pop)))
        file.write('Best fitness: %f' %(hof[0].fitness.values[0]))

        best_ind = tools.selBest(pop, 1)[0]
        acc = np.mean(best_ind == Yt)
        file.write("Accuracy of the best individual: %f" % acc)

        top10 = tools.selBest(pop, 10)
        vote_label = Helpers.voting(top10)
        acc = np.mean(vote_label == Yt)
        file.write("Accuracy of the 10%% population: %f" % acc)

        # Use the whole population
        vote_label = Helpers.voting(pop)
        acc = np.mean(vote_label == Yt)
        file.write("Accuracy of the population: %f" % acc)

    file.write("*****Final result*****")
    best_ind = tools.selBest(pop, 1)[0]
    acc = np.mean(best_ind == Yt)
    file.write("Accuracy of the best individual: %f" % acc)

    top10 = tools.selBest(pop, 10)
    vote_label = Helpers.voting(top10)
    acc = np.mean(vote_label == Yt)
    file.write("Accuracy of the 10%% population: %f" % acc)

    # Use the whole population
    vote_label = Helpers.voting(pop)
    acc = np.mean(vote_label == Yt)
    file.write("Accuracy of the population: %f" % acc)

if __name__ == '__main__':
    import sys

    run = 1 #int(sys.argv[1])
    random_seed = 1617 * run
    normalize = int(sys.argv[1]) == 1
    mutation_rate = float(sys.argv[2])/100
    full_init = int(sys.argv[3]) == 1
    dataset = sys.argv[4]

    # datasets = np.array(['SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a',
    #                      'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c',
    #                      'SURFd-w', 'SURFw-a', 'SURFw-c', 'SURFw-d',
                         # 'MNIST-USPS', 'USPS-MNIST', 'ICLEFc-i', 'ICLEFc-p',
                         # 'ICLEFi-c', 'ICLEFi-p', 'ICLEFp-c', 'ICLEFp-i'
                         # ])

    # datasets = np.array(['OfficeHomeArt-Product'])

    # datasets = np.array([ 'SURFc-d', 'SURFd-w', 'SURFw-a'])
    # for dataset in datasets:
    #     print('-------------------> %s <--------------------' %dataset)
    dir = '' #''/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/' + dataset
    source = np.genfromtxt(dir + "Source", delimiter=",")
    m = source.shape[1] - 1
    Xs = source[:, 0:m]
    Ys = np.ravel(source[:, m:m + 1])
    Ys = np.array([int(label) for label in Ys])

    target = np.genfromtxt(dir + "Target", delimiter=",")
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

    file = open(dataset+".txt", "w")

    file.write('----------------Setting------------------')
    file.write('Normalize: '+ str(normalize))
    file.write('Mutation rate: '+str(mutation_rate))
    file.write('Fully opposite initialize: '+str(full_init))
    file.write('----------------End setting------------------')
    file.write('')

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs, Ys)
    file.write('1NN accuracy: %f' %(np.mean(knn.predict(Xt)==Yt)))

    meda = MEDA.MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10, out=None)
    acc, ypre, list_acc = meda.fit_predict(Xs, Ys, Xt, Yt)
    file.write('MEDA accuracy: %f' % (acc))

    file.write('---------------GA-MEDA-----------------')
    evolve(Xs, Ys, Xt, Yt, file, mutation_rate, full_init)
