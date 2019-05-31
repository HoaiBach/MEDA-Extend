import random

import numpy as np
from deap import base, creator, tools
from sklearn import metrics, neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import GFK
import MEDA
import scipy.stats as stat


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
rate = 1.0

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
archive = []
archive_size_min = 10
random_rate = 0.5


def estimate_mu(_X1, _Y1, _X2, _Y2):
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


def fit_predict_beta(Beta):
    F = np.dot(K, Beta)
    Y_pseudo = np.argmax(F, axis=1) + 1
    Yt_pseu = Y_pseudo[ns:]

    mu = 0.5
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
    left = np.dot(rate*A + lamb * M + rho * L, K) \
           + rate * eta * np.eye(ns + nt, ns + nt)
    Beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

    # Now given the new beta, calculate the fitness
    SRM = np.linalg.norm(np.dot(YY.T - np.dot(Beta.T, K), A)) \
          + eta * np.linalg.multi_dot([Beta.T, K, Beta]).trace()
    MMD = np.linalg.multi_dot(
        [Beta.T, np.linalg.multi_dot([K, lamb * M + rho * L, K]), Beta]).trace()
    fitness = rate * SRM + MMD

    # Calcuate the accuracy
    F = np.dot(K, Beta)
    Y_pseudo = np.argmax(F, axis=1) + 1
    Ys_pseu = Y_pseudo[:ns]
    acc_s = np.mean(Ys_pseu == Ys)
    Yt_pseu = Y_pseudo[ns:]
    acc_t = np.mean(Yt_pseu == Yt)

    return Beta, fitness, MMD, SRM, acc_s, acc_t, Yt_pseu

def fit_predict_label(Yt_pseu):
    Yt_pseu = [Yt_pseu[index] for index in range(len(Yt_pseu))]
    Yt_pseu = np.array(Yt_pseu)
    mu = 0.5
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
    left = np.dot(rate*A + lamb * M + rho * L, K) \
           + rate*eta * np.eye(ns + nt, ns + nt)
    Beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

    # Now given the new beta, calculate the fitness
    SRM = np.linalg.norm(np.dot(YY.T - np.dot(Beta.T, K), A)) \
          + eta * np.linalg.multi_dot([Beta.T, K, Beta]).trace()
    MMD = np.linalg.multi_dot(
        [Beta.T, np.linalg.multi_dot([K, lamb * M + rho * L, K]), Beta]).trace()
    fitness = rate*SRM + MMD

    # Calcuate the accuracy
    F = np.dot(K, Beta)
    Y_pseudo = np.argmax(F, axis=1) + 1
    Ys_pseu = Y_pseudo[:ns]
    acc_s = np.mean(Ys_pseu == Ys)
    Yt_pseu = Y_pseudo[ns:]
    acc_t = np.mean(Yt_pseu == Yt)

    return Beta, fitness, MMD, SRM, acc_s, acc_t, Yt_pseu

def evolve(Xsource, Ysource, Xtarget, Ytarget):
    """
    Running GA algorithms, where each individual is a set of target pseudo labels.
    :return: the best solution of GAs.
    """
    global ns, nt, C, Xs, Ys, Xt, Yt, YY, K, A, e, M0, L, archive
    archive_label = []
    archive_beta = []
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

    # now run the meda
    meda = MEDA.meda()

    # build some matrices that are not changed
    K = kernel(kernel_type, X, X2=None, gamma=gamma)
    A = np.diagflat(np.vstack((np.ones((ns, 1)), np.zeros((nt, 1)))))
    e = np.vstack((1.0 / ns * np.ones((ns, 1)), -1.0 / nt * np.ones((nt, 1))))
    M0 = e * e.T * C
    L = laplacian_matrix(X.T, p)

    YY = np.zeros((ns, C))
    for c in range(1, C + 1):
        ind = np.where(Ys == c)
        YY[ind, c - 1] = 1
    YY = np.vstack((YY, np.zeros((nt, C))))

    # parameters for GA
    N_BIT = nt
    N_GEN = 10
    N_IND = 10
    MUTATION_RATE = 1.0 / N_BIT
    CXPB = 0.2
    MPB = 0.8

    pos_min = 1
    pos_max = C

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Ind", list, fitness=creator.FitnessMin, beta=list)
    creator.create("Pop", list)

    # for creating the population
    toolbox.register("bit", random.randint, a=pos_min, b=pos_max)
    toolbox.register("ind", tools.initRepeat, creator.Ind, toolbox.bit, n=N_BIT)
    toolbox.register("pop", tools.initRepeat, creator.Pop, toolbox.ind)
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
    classifiers.append(SVC(kernel="linear", C=0.025, random_state=np.random.randint(2 ** 10)))
    classifiers.append(GaussianProcessClassifier(1.0 * RBF(1.0), random_state=np.random.randint(2 ** 10)))
    classifiers.append(KNeighborsClassifier(3))
    classifiers.append(SVC(kernel="rbf", C=1, gamma=2, random_state=np.random.randint(2 ** 10)))
    classifiers.append(DecisionTreeClassifier(max_depth=5, random_state=np.random.randint(2 ** 10)))
    classifiers.append(KNeighborsClassifier(5))
    classifiers.append(GaussianNB())
    classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10, random_state=np.random.randint(2 ** 10)))
    classifiers.append(AdaBoostClassifier(random_state=np.random.randint(2 ** 10)))

    step = N_IND / len(classifiers)
    for ind_index, classifier in enumerate(classifiers):
        classifier.fit(Xs, Ys)
        Yt_pseu = classifier.predict(Xt)
        for bit_idex, value in enumerate(Yt_pseu):
            pop[ind_index * step][bit_idex] = value

    # for index, value in enumerate(Yt):
    #     pop[len(pop)-1][index] = Yt[index]

    # now initialize the beta
    # Beta, fitness, MMD, SRM, acc_s, acc_t, Yt_pseu
    for ind in pop:
        beta, fitness, _, _, _, _, Yt_pseu  = fit_predict_label(ind)
        ind.beta = beta
        ind.fitness.values = fitness,
        for idx in range(len(ind)):
            ind[idx] = Yt_pseu[idx]

    hof = tools.HallOfFame(maxsize=1)
    hof.update(pop)

    for g in range(N_GEN):
        # print("=========== Iteration %d ===========" % g)
        # selection
        offspring_label = toolbox.select(pop, len(pop))
        offspring_label = map(toolbox.clone, offspring_label)

        # applying crossover
        for c1, c2 in zip(offspring_label[::2], offspring_label[1::2]):
            if np.random.rand() < CXPB:
                toolbox.crossover(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # applying mutation
        for mutant in offspring_label:
            if np.random.rand() < MPB:
                # retain the fitness value of parent to children
                toolbox.mutate(mutant)
                del mutant.fitness.values

        inv_inds = [ind for ind in offspring_label if not ind.fitness.valid]
        for inv_ind in inv_inds:
            beta, fitness, _, _, _, _, Yt_pseu = fit_predict_label(inv_ind)
            inv_ind.beta = beta
            inv_ind.fitness.values = fitness,
            for idx in range(len(inv_ind)):
                inv_ind[idx] = Yt_pseu[idx]

        # Now using beta phase to evolve the current pop
        offspring_beta = map(toolbox.clone, pop)
        for ind in offspring_beta:
            beta, fitness, _, _, _, _, Yt_pseu = fit_predict_beta(ind.beta)
            ind.beta = beta
            ind.fitness.values = fitness,
            for idx in range(len(inv_ind)):
                inv_ind[idx] = Yt_pseu[idx]

        for idx in range(len(pop)):
            ind = pop[idx]
            ind_label = offspring_label[idx]
            ind_beta = offspring_beta[idx]
            if ind.fitness.values <= ind_label.fitness.values and ind.fitness.values <= ind_beta.fitness.values:
                # if both approaches do not improve the fitness, indicate a local optima
                # create a new one
                new_label = re_initialize()
                new_beta, new_fitness, _, _, _, _, new_label = fit_predict_label(new_label)
                # store the current one to archive
                archive_beta.append(ind.beta)
                pseudo_labels = [ind[ins_idx] for ins_idx in range(len(ind))]
                archive_label.append(pseudo_labels)
                # assign the new one to the current individual
                for bit_index in range(len(new_label)):
                    pop[idx][bit_index] = new_label[bit_index]
                pop[idx].beta = new_beta
                pop[idx].fitness.values = new_fitness,
            elif ind.fitness.values > ind_label.fitness.values and ind_beta.fitness.values > ind_label.fitness.values:
                # the beta step improve the fitness
                pop[idx] = toolbox.clone(ind_label)
            elif ind.fitness.values > ind_beta.fitness.values and ind_label.fitness.values > ind_beta.fitness.values:
                pop[idx] = toolbox.clone(ind_beta)

        hof.update(pop)
        best_ind = tools.selBest(pop, 1)[0]
        # print("Best fitness %f " % best_ind.fitness.values)

    # print("=========== Final result============")
    best_ind = tools.selBest(pop, 1)[0]
    _, _, _, _, acc_s, acc_t, Yt_pseu = fit_predict_label(best_ind)
    print("Accuracy of the best individual: source - %f target - %f " %(acc_s, acc_t))

    labels = []
    for ind in pop:
        _, _, _, _, _, _, Yt_pseu = fit_predict_label(ind)
        labels.append(Yt_pseu)
    labels = np.array(labels)
    vote_label = voting(labels)
    acc = np.mean(vote_label == Yt)
    print("Accuracy of the population: %f" % acc)

    # Use the archive of beta
    labels = []
    for beta in archive_beta:
        _, _, _, _, _, _, Yt_pseu = fit_predict_beta(beta)
        labels.append(Yt_pseu)
    labels = np.array(labels)
    vote_label = voting(labels)
    acc = np.mean(vote_label == Yt)
    print("Accuracy of the archive beta: %f" % acc)

    # Use the archive of label
    labels = []
    for label in archive_label:
        _, _, _, _, _, _, Yt_pseu = fit_predict_label(label)
        labels.append(Yt_pseu)
    labels = np.array(labels)
    vote_label = voting(labels)
    acc = np.mean(vote_label == Yt)
    print("Accuracy of the archive label: %f" % acc)


def voting(set_labels):
    vote_label = []
    ins_labels = []
    for labels in set_labels:
        tmp_labels = [labels[index] for index in range(len(labels))]
        ins_labels.append(tmp_labels)
    ins_labels = np.array(ins_labels)

    for m_index in range(len(ins_labels[0])):
        counts = np.bincount(ins_labels[:, m_index])
        label = np.argmax(counts)
        vote_label.append(label)

    return vote_label


def index_duplicate(pop):
    '''
    :param pop: population
    :return: indices of the duplicated solutions
    '''
    seen = []
    idx_dup = []
    for idx, ind in enumerate(pop):
        if check_contain(ind, seen):
            idx_dup.append(idx)
        else:
            seen.append(ind)
    return idx_dup


def check_contain(oneD, twoD):
    '''
    Check whethere oneD is a row in twoD
    :param oneD: an 1D array
    :param twoD: an 2D array
    :return:
    '''
    if len(twoD) == 0:
        return False
    return np.max([np.min(oneD == row) for row in twoD])


def re_initialize():
    new_ind = toolbox.ind()
    if len(archive) < archive_size_min or np.random.rand() <= random_rate:
        return new_ind
    else:
        pseu_labels = []
        for ins_index in range(nt):
            list_labels = [archive[i][ins_index] for i in range(len(archive))]
            ins_label = np.random.choice(np.unique(list_labels))
            pseu_labels.append(ins_label)
        for index, label in enumerate(pseu_labels):
            new_ind[index] = label
        return new_ind


def is_in(array, matrix):
    """
    Check wherether an array matches a row in the matrix
    Both represented by a numpy array
    :param array:
    :param matrix:
    :return:
    """
    if len(matrix) == 0:
        return False
    else:
        return np.any(np.sum(np.abs(matrix - array), axis=1) == 0)


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
        sim[i] = [sim[i][j] / (S[i] * S[j]) ** 0.5 for j in range(len(sim))]

    L = np.identity(len(sim)) - sim
    return L


if __name__ == '__main__':
    import sys

    run = int(sys.argv[1])
    random_seed = 1617 * run

    datasets = np.array(['SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a',
                         'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c',
                         'SURFd-w', 'SURFw-a', 'SURFw-c', 'SURFw-d',
                         'MNIST-USPS', 'USPS-MNIST', 'ICLEFc-i', 'ICLEFc-p',
                         'ICLEFi-c', 'ICLEFi-p', 'ICLEFp-c', 'ICLEFp-i',
                         'Office31amazon-dslr', 'Office31dslr-webcam', 'Office31webcam-amazon',
                         'OfficeHomeArt-Clipart', 'OfficeHomeClipart-Product',
                         'OfficeHomeProduct-RealWorld', 'OfficeHomeRealWorld-Art'])

    # datasets = np.array(['SURFa-c',
    #                      'SURFc-d',
    #                      'SURFd-w', 'SURFw-a',
    #                      'ICLEFc-i',
    #                      'ICLEFi-p', 'ICLEFp-c'])

    datasets = np.array(['ICLEFc-i'])
                                     
    for dataset in datasets:
        print('-------------------> %s <--------------------' % dataset)
        dir = '/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/' + dataset
        source = np.genfromtxt(dir + "/Source", delimiter=",")
        m = source.shape[1] - 1
        Xs = source[:, 0:m]
        Ys = np.ravel(source[:, m:m + 1])
        Ys = np.array([int(label) for label in Ys])

        target = np.genfromtxt(dir + "/Target", delimiter=",")
        Xt = target[:, 0:m]
        Yt = np.ravel(target[:, m:m + 1])
        Yt = np.array([int(label) for label in Yt])

        C = len(np.unique(Ys))
        if C > np.max(Ys):
            Ys = Ys + 1
            Yt = Yt + 1

        Xs = Xs.T
        Xs /= np.sum(Xs, axis=0)
        Xs = Xs
        Xs = stat.zscore(Xs, axis=1)

        Xt /= np.sum(Xt, axis=0)
        Xt = stat.zscore(Xt, axis=1)

        #
        # Xt = Xt.T
        # Xt /= np.linalg.norm(Xt, axis=0)
        # Xt = Xt.T

        # X = np.hstack((Xs.T, Xt.T))
        # X /= np.linalg.norm(X, axis=0)
        # Xs = X[:, :len(Xs)].T
        # Xt = X[:, len(Xs):].T

        knn = KNeighborsClassifier(1).fit(Xs,Ys)
        print(np.mean(knn.predict(Xt)==Yt))

        np.random.seed(random_seed)
        random.seed(random_seed)

        evolve(Xs, Ys, Xt, Yt)
