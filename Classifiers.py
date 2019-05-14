import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

datasets = np.array(['GasSensor1-4', 'GasSensor1-2', 'GasSensor1-3',
                     'GasSensor1-5', 'GasSensor1-6', 'GasSensor1-7',
                     'GasSensor1-8', 'GasSensor1-9', 'GasSensor1-10',
                     'SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a',
                     'SURFc-d', 'SURFc-w', 'SURFd-a', 'SURFd-c',
                     'SURFd-w', 'SURFw-a', 'SURFw-c', 'SURFw-d',
                     'MNIST-USPS', 'USPS-MNIST'])

classifiers = list([])
classifiers.append(KNeighborsClassifier(1))
classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10, random_state=np.random.randint(2 ** 10)))
classifiers.append(SVC(kernel="rbf", C=1, gamma=2, random_state=np.random.randint(2 ** 10)))

names = list(["1NN", "RF", "SVM"])

for dataset in datasets:
    source = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Source",
                           delimiter=",")
    m = source.shape[1] - 1
    Xs = source[:, 0:m]
    Ys = np.ravel(source[:, m:m + 1])
    Ys = np.array([int(label) for label in Ys])

    target = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Target",
                           delimiter=",")
    Xt = target[:, 0:m]
    Yt = np.ravel(target[:, m:m + 1])
    Yt = np.array([int(label) for label in Yt])

    for index, classifier in enumerate(classifiers):
        classifier.fit(Xs, Ys)
        acc = classifier.score(Xt, Yt)
        file = open("/home/nguyenhoai2/Grid/results/R-MEDA/" + dataset + "/" + names[index] + ".txt", "w")
        file.write(str(acc))
        file.close()
