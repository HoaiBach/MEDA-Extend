import numpy as np
import Core


def fitness_function_test(Beta, Yt_pseu):
    '''
    Calculate the fitness function of the matrix A
    :param A: A_row * A_col, defined in Core
    :return:
    '''
    # estimate the new label
    Ytest = np.copy(Core.YY)
    for c in range(1, Core.C + 1):
        yy = Yt_pseu == c
        inds = np.where(yy == True)
        inds = [item + Core.ns for item in inds]
        Ytest[inds, c - 1] = 1

    # now build M
    N = 0
    for c in range(1, Core.C + 1):
        e = np.zeros((Core.n, 1))
        tt = Core.Ys == c
        e[np.where(tt == True)] = 1.0 / len(Core.Ys[np.where(Core.Ys == c)])
        yy = Yt_pseu == c
        ind = np.where(yy == True)
        inds = [item + Core.ns for item in ind]
        if len(Yt_pseu[np.where(Yt_pseu == c)]) != 0:
            e[tuple(inds)] = -1.0 / len(Yt_pseu[np.where(Yt_pseu == c)])
        else:
            e[np.isinf(e)] = 0
        N = N + np.dot(e, e.T)
    M = 0.5*Core.M0 + 0.5*N
    M = M / np.linalg.norm(M, 'fro')

    SRM = np.linalg.norm(np.dot(Ytest.T - np.dot(Beta.T, Core.K), Core.A)) \
          + Core.eta * np.linalg.multi_dot([Beta.T, Core.K, Beta]).trace()
    MMD = Core.lamb * np.linalg.multi_dot([Beta.T, np.linalg.multi_dot([Core.K, M, Core.K]), Beta]).trace()

    return SRM+MMD


def fitness_function(Beta):
    '''
    Calculate the fitness function of the matrix A
    :param A: A_row * A_col, defined in Core
    :return:
    '''
    # estimate the new label
    Ytest = np.copy(Core.YY)
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


def refine(beta):
    # estimate the new label
    F = np.dot(Core.K, beta)
    Cls = np.argmax(F, axis=1) + 1
    Cls = Cls[Core.ns:]

    # now build M
    N = 0
    for c in range(1, Core.C + 1):
        e = np.zeros((Core.n, 1))
        tt = Core.Ys == c
        e[np.where(tt == True)] = 1.0 / len(Core.Ys[np.where(Core.Ys == c)])
        yy = Cls == c
        ind = np.where(yy == True)
        inds = [item + Core.ns for item in ind]
        if len(Cls[np.where(Cls == c)]) != 0:
            e[tuple(inds)] = -1.0 / len(Cls[np.where(Cls == c)])
        else:
            e[np.isinf(e)] = 0
        N = N + np.dot(e, e.T)
    M = 0.5 * Core.M0 + 0.5 * N
    M = M / np.linalg.norm(M, 'fro')

    left = np.dot(Core.A + Core.lamb * M, Core.K) + Core.eta * np.eye(Core.n, Core.n)
    new_beta = np.dot(np.linalg.inv(left), np.dot(Core.A, Core.YY))

    return new_beta


def fitness_cheat(beta):
    beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
    F = np.dot(Core.K, beta)
    Cls = np.argmax(F, axis=1) + 1
    Cls = Cls[Core.ns:]
    Core.Yt_pseu = Cls
    acc = np.mean(Core.Yt_pseu == Core.Yt)
    return 1-acc


if __name__ == "__main__":
    v1 = np.asarray([-3, -2, 6])
    v2 = np.asarray([4, 5, -8])
