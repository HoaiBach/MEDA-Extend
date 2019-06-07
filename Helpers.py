import numpy as np
import scipy.stats as stat
from sklearn import neighbors

def normalize_data(Xs, Xt):
    Xs = Xs.T
    Xs /= np.sum(Xs, axis=0)
    Xs = Xs.T
    Xs = stat.zscore(Xs)

    Xt = Xt.T
    Xt /= np.sum(Xt, axis=0)
    Xt = Xt.T
    Xt = stat.zscore(Xt)

    Xs = np.nan_to_num(Xs)
    Xt = np.nan_to_num(Xt)

    return Xs, Xt

def distance(ind1, ind2):
    '''
    return manhatan distance between 2 individuals
    :param ind1:
    :param ind2:
    :return:
    '''
    count_dif = 0
    for index, value1 in enumerate(ind1):
        value2 = ind2[index]
        if value1 != value2:
            count_dif += 1
    return float(count_dif)/len(ind1)

def pop_distance(pop):
    '''
    return the average distance between individuals in pop
    :param pop:
    :return:
    '''
    sum_dis = 0.0
    for i in range(len(pop)-1):
        for j in range(i+1, len(pop)):
            sum_dis = sum_dis+distance(pop[i], pop[j])
    return sum_dis/(len(pop)*(len(pop)-1)/2.0)

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
        return np.any(np.sum(np.abs(matrix-array), axis=1) == 0)

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

def opposite_init(pop, min_pos, max_pos, full_init):
    '''
    Try to initialize the population by an oppositing position
    in this case, we assume that the first half of the pop has been
    initialize,
    :param pop: the population where the first half was initialized
    :param min_pos: minimum pos value
    :param max_pos: maximum pos value
    :return: none, changes are made inside pop
    '''
    # pop is initialized from 0 to init_upto
    if full_init:
        for ind_index in range(0, len(pop), max_pos):
            if ind_index+max_pos < len(pop):
                inited = pop[ind_index]
                for pos_index, value in enumerate(inited):
                    values = [other for other in range(min_pos, max_pos+1) if other != value]
                    np.random.shuffle(values)
                    for add_index in range(1, max_pos-1):
                        pop[ind_index+add_index][pos_index] = values[add_index-1]
    else:
        for i in range(0, len(pop), 3):
            if i+2 < len(pop):
                inited = pop[i]
                to_init1 = pop[i+1]
                to_init2 = pop[i+2]
                for index, value in enumerate(inited):
                    rand_value1 = value
                    while rand_value1 == value:
                        rand_value1 = np.random.randint(min_pos, max_pos+1)
                    to_init1[index] = rand_value1
                    rand_value2 = value
                    while rand_value2 == value or rand_value2 == rand_value1:
                        rand_value2 = np.random.randint(min_pos, max_pos+1)
                    to_init2[index] = rand_value2
            else:
                break
