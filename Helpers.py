import numpy as np
import scipy.stats as stat


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