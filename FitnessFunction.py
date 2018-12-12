import numpy as np
import Core

def fitness_function(Beta):
    '''
    Calculate the fitness function of the matrix A
    :param A: A_row * A_col, defined in Core
    :return:
    '''
    # estimate the new label
    Ytest = np.copy(Core.Yt_pseu)
    for c in range(1, Core.C + 1):
        yy = Core.Yt_pseu == c
        inds = np.where(yy == True)
        inds = [item + Core.ns for item in inds]
        Ytest[inds, c - 1] = 1

    # now build M
    N = 0
    for c in range(1, Core.C + 1):
        e = np.zeros((Core.n, 1))
        tt = Core.Ys == c
        e[np.where(tt == True)] = 1.0 / len(Core.Ys[np.where(Core.Ys == c)])
        yy = Core.Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + Core.ns for item in ind]
        if len(Core.Yt_pseu[np.where(Core.Yt_pseu == c)]) != 0:
            e[tuple(inds)] = -1.0 / len(Core.Yt_pseu[np.where(Core.Yt_pseu == c)])
        else:
            e[np.isinf(e)] = 0
        N = N + np.dot(e, e.T)
    M = 0.5*Core.M0 + 0.5*N
    M = M / np.linalg.norm(M, 'fro')

    SRM = np.linalg.norm(np.dot(Ytest.T - np.dot(Beta.T, Core.K), Core.A)) \
          + Core.eta * np.linalg.multi_dot([Beta.T, Core.K, Beta]).trace()
    MMD = Core.lamb * np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([Core.K, M, Core.K]), Beta]).trace()

    return SRM+MMD


if __name__ == "__main__":
    v1 = np.asarray([-3, -2, 6])
    v2 = np.asarray([4, 5, -8])
