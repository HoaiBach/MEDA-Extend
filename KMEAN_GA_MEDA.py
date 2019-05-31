import random

import numpy as np
from deap import base, creator, tools
from sklearn import svm, metrics, neighbors, cluster
from scipy.spatial.distance import pdist,squareform

import GFK

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import Directory as Dir


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
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    # if one of the source or taget dose not have the label
    # return the worse case
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


kernel_type = 'rbf'
dim = 20
lamb = 10
rho = 1.0
eta = 0.1
p = 10
gamma = 0.5
# T = 10

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
L = 0
sim = 0

toolbox = base.Toolbox()
archive = []
archive_size_min = 10
random_rate = 0.5

def manifold(Yt_input):
    Yt_pseu = np.array(np.copy(Yt_input))
    Y_pseu = np.append(Ys, Yt_pseu)
    one_hot = np.zeros((len(Y_pseu), C))
    for c in range(1, C+1):
        tt = Y_pseu == c
        for index in np.where(tt == True)[0]:
            one_hot[index][c-1] = 1
    return np.sum([sim[i][j]*np.linalg.norm(Y_pseu[i]-Y_pseu[j])
                   for i in range(len(sim)) for j in range(len(sim))])


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


def meda(Yt_init, T):
    """
    Calculate the fitness function of a label array of target instances
    :param Yt_pseu: the pseudolabel of target instances, which are in a chromosome.
    :return:
    """
    # Based on the Yt_pseu, calculate the optimal beta.
    Yt_pseu = np.array(np.copy(Yt_init))
    for t in range(1, T+1):
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
        Beta = np.dot(np.linalg.inv(left), np.dot(A, YY))

        Ytest = np.copy(YY)
        for c in range(1, C+1):
            yy = Yt_pseu == c
            inds = np.where(yy == True)
            inds = [item + ns for item in inds]
            Ytest[inds, c-1] = 1

        # Now given the new beta, calculate the fitness
        # For testing only
        # SRM = np.linalg.norm(np.dot(Ytest.T - np.dot(Beta.T, K), A)) \
        #     + eta * np.linalg.multi_dot([Beta.T, K, Beta]).trace()
        # MMD = np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([K, lamb * M + rho * L, K]), Beta]).trace()
        # fitness = SRM + MMD

        F = np.dot(K, Beta)
        Y_pseu = np.argmax(F, axis=1) + 1
        Yt_pseu = Y_pseu[ns:]
    return Yt_pseu


def Kmean_MEDA(Xsource, Ysource, Xtarget, Ytarget):
    '''
    :param Xs: a numpy array size ns*m
    :param Ys: a numpy array size 1*ns
    :param Xt: a numpy array size nt*m
    :param Yt: a numpy array size 1*nt
    :return: accuracy
    '''
    global ns, nt, C, Xs, Ys, Xt, Yt, YY, K, A, e, M0, L, archive, sim
    Xs = Xsource
    Ys = Ysource
    Xt = Xtarget
    Yt = Ytarget
    ns, nt, m = Xs.shape[0], Xt.shape[0], Xs.shape[1]
    C = len(np.unique(Ys))

    # Transform data using gfk
    gfk = GFK.GFK(dim=dim)
    _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
    Xs_new, Xt_new = Xs_new.T, Xt_new.T
    X = np.hstack((Xs_new, Xt_new))
    X /= np.linalg.norm(X, axis=0)
    Xs = X[:, :ns].T
    Xt = X[:, ns:].T
    X = X.T

    # perform clustering
    kmeans = cluster.KMeans(n_clusters=C, random_state=0, max_iter=200).fit(X)
    source_cluster_kmeans = kmeans.labels_[:ns]
    # dbscan = cluster.DBSCAN(eps=3, min_samples=3).fit(X)
    # source_cluster_dbscan = dbscan.labels_[:ns]
    aggc = cluster.AgglomerativeClustering(n_clusters=C, linkage='complete').fit(X)
    source_cluster_aggc = aggc.labels_[:ns]
    print 'test'


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

def simliarity_matrix(data, k):
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
    return sim


def laplacian_matrix(data, k):
    """
    :param data: containing data points,
    :param k: the number of neighbors considered (this distance metric is cosine,
    and the weights are measured by cosine)
    :return:
    """
    sim = simliarity_matrix(data, k)

    S = [np.sum(row) for row in sim]

    for i in range(len(sim)):
        sim[i] = [sim[i][j]/(S[i]*S[j])**0.5 for j in range(len(sim))]

    L = np.identity(len(sim)) - sim
    return L


if __name__ == '__main__':
    import sys

    run = int(sys.argv[1])
    random_seed = 1617 * run

    datasets = np.array(['SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a',
                         'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c',
                         'SURFd-w', 'SURFw-a', 'SURFw-c', 'SURFw-d',
                         'ICLEFc-i', 'ICLEFi-p', 'ICLEFp-c',
                         'ICLEFc-p', 'ICLEFp-i', 'ICLEFi-c',
                         'MNIST-USPS', 'USPS-MNIST'])

    # datasets = np.array(['Office31amazon-dslr', 'Office31dslr-webcam', 'Office31webcam-amazon',
    #                      'OfficeHomeArt-Clipart','OfficeHomeClipart-Product', 'OfficeHomeProduct-RealWorld',
    #                      'OfficeHomeRealWorld-Art'])
    # datasets = np.array(['GasSensor1-2', 'GasSensor1-3', 'GasSensor1-4', 'GasSensor1-5', 'GasSensor1-6', 'GasSensor1-7',
    #                      'GasSensor1-8', 'GasSensor1-9'])
    datasets = np.array(['SURFd-w'])
    for dataset in datasets:
        print("==========%s=========" %dataset)
        dir = Dir.dir + dataset
        source = np.genfromtxt(dir + "/Source", delimiter=",")
        m = source.shape[1] - 1
        Xs = source[:, 0:m]
        Ys = np.ravel(source[:, m:m + 1])
        Ys = np.array([int(label) for label in Ys])

        target = np.genfromtxt(dir + "/Target", delimiter=",")
        Xt = target[:, 0:m]
        Yt = np.ravel(target[:, m:m + 1])
        Yt = np.array([int(label) for label in Yt])

        # require all the labels start from 1
        if np.min(np.unique(Ys)) == 0:
            Ys = Ys+1
            Yt = Yt+1

        np.random.seed(random_seed)
        random.seed(random_seed)

        Kmean_MEDA(Xs, Ys, Xt, Yt)

