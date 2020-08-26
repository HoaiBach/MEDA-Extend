
"""
Multi-objective MEDA:
1st objective: discrepancy
2nd objective: manifold
"""

# from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.evaluator import MultiprocessEvaluator, SparkEvaluator
from jmetal.lab.visualization import Plot
from jmetal.util.ranking import FastNonDominatedRanking

import MultiTransfer as mt
from jmetal.util.termination_criterion import StoppingByEvaluations

from sklearn import metrics, neighbors
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import Helpers
import GFK
import MEDA
import time
import numpy as np
import random

import Paras
from NSGAII_MEDA import NSGAII_MEDA

if __name__ == '__main__':
    import sys,os

    # run = int(sys.argv[1])
    # random_seed = 1617 * run
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    # normalize = int(sys.argv[2]) == 1
    # full_init = int(sys.argv[4]) == 1
    # dim = int(sys.argv[5])
    # eta = float(sys.argv[6])/100.0
    #
    # source = np.genfromtxt("Source", delimiter=",")
    # m = source.shape[1] - 1
    # Xs = source[:, 0:m]
    # Ys = np.ravel(source[:, m:m + 1])
    # Ys = np.array([int(label) for label in Ys])
    #
    # target = np.genfromtxt("Target", delimiter=",")
    # Xt = target[:, 0:m]
    # Yt = np.ravel(target[:, m:m + 1])
    # Yt = np.array([int(label) for label in Yt])
    #
    # if normalize:
    #     Xs, Xt = Pre.normalize_data(Xs, Xt)
    #
    # C = len(np.unique(Ys))
    # if C > np.max(Ys):
    #     Ys = Ys + 1
    #     Yt = Yt + 1
    #
    # file = open(str(run)+".txt", "w")
    #
    # file.write('----------------Setting------------------\n')
    # file.write('Pop size: '+str(N_IND)+'\n')
    # file.write('Max iterations: '+str(N_GEN)+'\n')
    # file.write('Normalize: '+ str(normalize)+'\n')
    # file.write('Mutation rate: '+str(mutation_rate)+'\n')
    # file.write('Fully opposite initialize: '+str(full_init)+'\n')
    # file.write('GFK dim: '+str(dim)+'\n')
    # file.write('Eta: '+str(eta)+'\n')
    # file.write('----------------End setting------------------'+'\n')
    #
    # knn = KNeighborsClassifier(n_neighbors=1)
    # knn.fit(Xs, Ys)
    # file.write('1NN accuracy: %f' %(np.mean(knn.predict(Xt)==Yt))+'\n')
    #
    # start = time.time()
    # meda = MEDA.MEDA(kernel_type='rbf', dim=dim, lamb=10, rho=1.0, eta=eta, p=10, gamma=0.5, T=10, out=None)
    # acc, ypre, list_acc = meda.fit_predict(Xs, Ys, Xt, Yt)
    # end = time.time()
    # exe_time = end - start
    # file.write('MEDA accuracy: %f' % (acc)+'\n')
    # file.write('MEDA time: %f' % (exe_time)+'\n')
    #
    # file.write('---------------GA-MEDA-----------------'+'\n')
    # evolve(Xs, Ys, Xt, Yt, file, mutation_rate, full_init, dim_p=dim, eta_p=eta)
    random.seed(17)
    np.random.seed(17)

    problem = mt.MultiTransferProblem(normalize=False, gfk_dim=20)
    no_evaluations = Paras.N_GEN*Paras.N_IND
    algorithm = NSGAII_MEDA(
        problem=problem,
        population_size=Paras.N_IND,
        offspring_population_size=Paras.N_IND,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=IntegerSBXCrossover(probability=0.8, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=no_evaluations)
    )
    progress_bar = ProgressBarObserver(max=no_evaluations)
    algorithm.observable.register(progress_bar)
    algorithm.run()
    solutions = algorithm.get_result()
    non_dominated_ranking = FastNonDominatedRanking(algorithm.dominance_comparator)
    non_dominated_ranking.compute_ranking(solutions)
    non_dominated_sols = non_dominated_ranking.get_nondominated()
    plot_front = Plot(title='Pareto front approximation', axis_labels=['Dis', 'Mani'])
    plot_front.plot(non_dominated_sols, label='NSGAII-ZDT1')

    list_labels = []
    for sol in non_dominated_sols:
        list_labels.append(sol.variables)
        dis = sol.objectives[0]
        man = sol.objectives[1]
        # train_acc = sol.objectives[2]
        Yt_sol = sol.variables
        acc = (np.sum(Yt_sol == problem.Yt)+0.0)/len(Yt_sol)
        print("Dis: %.5f, man: %.5f, acc: %.5f" %(dis, man, acc))
        # print("Dis: %.5f, man: %.5f, train acc: %.5f, acc: %.5f" % (dis, man, train_acc ,acc))

    vote_label = Helpers.voting(list_labels)
    acc = (np.sum(Yt_sol == problem.Yt)+0.0)/len(Yt_sol)
    print("Voting acc: %.5f" % acc)

