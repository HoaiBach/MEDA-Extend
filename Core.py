import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import GFK
import Utility

src_data = np.genfromtxt("data/Source", delimiter=",")
no_features = len(src_data[0]) - 1
Xs = src_data[:, 0:no_features]
Ys = np.ravel(src_data[:, no_features:no_features + 1])
Ys = np.array([int(label) for label in Ys])
C = len(np.unique(Ys))

tar_data = np.genfromtxt("data/Target", delimiter=",")
Xt = tar_data[:, 0:no_features]
Yt = np.ravel(tar_data[:, no_features:no_features + 1])
Yt = np.array([int(label) for label in Yt])

ns, nt = Xs.shape[0], Xt.shape[0]
n = ns + nt

YY = np.zeros((ns, C))
for c in range(1, C + 1):
    ind = np.where(Ys == c)
    YY[ind, c - 1] = 1
YY = np.vstack((YY, np.zeros((nt, C))))

gfk = GFK.GFK(dim=20)
_, Xs_new, Xt_new = gfk.fit(Xs, Xt)
Xs_new /= np.linalg.norm(Xs_new, axis=1)[:, None]
Xt_new /= np.linalg.norm(Xt_new, axis=1)[:, None]

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(Xs_new, Ys)
Yt_pseu = classifier.predict(Xt_new)

e = np.vstack((1.0 / ns * np.ones((ns, 1)), -1.0 / nt * np.ones((nt, 1))))
M0 = e * e.T * C

X = np.vstack((Xs_new, Xt_new))
K = Utility.kernel(ker='rbf', X=X.T, X2=None, gamma=0.5)
A = np.diagflat(np.vstack((np.ones((ns, 1)), np.zeros((nt, 1)))))

# parameters
lamb = 10
rho = 1.0
eta = 0.1
