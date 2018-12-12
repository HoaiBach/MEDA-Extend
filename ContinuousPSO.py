'''
Created on 24/09/2018

@author: nguyenhoai2
'''
import operator
import random
from deap import base
from deap import creator
from deap import tools
import Core
import FitnessFunction
import time
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math
import numpy as np

# Setting for SBPSO
NBIT = Core.no_features
NGEN = 100
NPART = NBIT if NBIT < 100 else 100
is_low = 0
is_up = 10.0/NBIT
ustks_low = NGEN/100.0
ustks_up = 8*NGEN/100.0
pg_rate = 2.0
threshold = 0.5

i_stick = is_up
i_gbest = (1-i_stick)/(pg_rate+1)
i_pbest = pg_rate * i_gbest
ustks = ustks_low

TEST = False
WRITE_OUT = False

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, stk=list, best=None)


def generate(size):
    part = creator.Particle(1 if random.uniform(0, 1) > threshold else 0 for _ in range(size))
    part.stk = [1 for _ in range(size)]
    return part


def update_particle(part, best):
    # find flipping probability
    stick_part = map(lambda x: i_stick*(1-x), part.stk)
    diff_pbest = map(operator.abs, map(operator.sub, part.best, part))
    pbest_part = map(lambda x: i_pbest * x, diff_pbest)
    diff_gbest = map(operator.abs, map(operator.sub, best, part))
    gbest_part = map(lambda x: i_gbest * x, diff_gbest)

    if TEST:
        print("Stick", stick_part)
        print("diff_pbest", pbest_part)
        print("diff_gbest", gbest_part)

    flipping = map(operator.add, stick_part, map(operator.add, pbest_part, gbest_part))

    # update the position -> update the stickiness
    for i, prob in enumerate(flipping):
        if random.random() < prob:
            # flip, change stickness back to 0:
            part[i] = 1 - part[i]
            part.stk[i] = 1
        else:
            # if the bit is not flipped, update stickiness
            part.stk[i] = max(0, part.stk[i] - 1.0/ustks)


def evaluate(particle):
    # Find all selected features
    indices = [index for index, entry in enumerate(particle) if entry == 1.0]
    # Build new dataset with selected features
    src_feature = Core.Xs[:, indices]
    tar_feature = Core.Xt[:, indices]

    # return FitnessFunction.fitness_confuse(src_feature, Core.src_label, tar_feature, Core.classifier),
    # return FitnessFunction.fitness_function(src_feature=src_feature, src_label=Core.src_label,
    #                                         tar_feature=tar_feature, classifier=Core.classifier)[0],
    return FitnessFunction.fitness_function_framework(src_feature=src_feature, src_label=Core.Ys,
                                                      classifier=Core.classifier, tar_feature=tar_feature)[0],


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=NBIT)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update_particle)
toolbox.register("evaluate", evaluate)


def normalize_weight():
    # Make sure that all the weights are not 0, so all components are evaluated
    FitnessFunction.srcWeight = 0.3
    FitnessFunction.margWeight = 0.3
    FitnessFunction.tarWeight = 0.3

    src_err, diff_marg, tar_err = \
        FitnessFunction.domain_differece(src_feature=Core.Xs, src_label=Core.Ys,
                                         classifier=Core.classifier, tar_feature=Core.Xt)

    if diff_marg == 0:
        FitnessFunction.margWeight = 0
    else:
        FitnessFunction.margWeight = 1.0 / abs(diff_marg)

    if tar_err == 0:
        FitnessFunction.tarWeight = 0
    else:
        FitnessFunction.tarWeight = 1.0 / abs(tar_err)

    if src_err == 0:
        FitnessFunction.srcWeight = 0
    else:
        FitnessFunction.srcWeight = 1.0/abs(src_err)

        def normalize_weight():
            # Make sure that all the weights are not 0, so all components are evaluated
            FitnessFunction.srcWeight = 0.3
            FitnessFunction.margWeight = 0.3
            FitnessFunction.tarWeight = 0.3

            src_err, diff_marg, tar_err = \
                FitnessFunction.domain_differece(src_feature=Core.Xs, src_label=Core.Ys,
                                                 classifier=Core.classifier, tar_feature=Core.Xt)

            if diff_marg == 0:
                FitnessFunction.margWeight = 0
            else:
                FitnessFunction.margWeight = 1.0 / abs(diff_marg)

            if tar_err == 0:
                FitnessFunction.tarWeight = 0
            else:
                FitnessFunction.tarWeight = 1.0 / abs(tar_err)

            if src_err == 0:
                FitnessFunction.srcWeight = 0
            else:
                FitnessFunction.srcWeight = 1.0 / abs(src_err)


def opposite_weight():
    # Make sure that all the weights are not 0, so all components are evaluated
    FitnessFunction.srcWeight = 0.3
    FitnessFunction.margWeight = 0.3
    FitnessFunction.tarWeight = 0.3

    src_err, diff_marg, tar_err = \
        FitnessFunction.domain_differece(src_feature=Core.Xs, src_label=Core.Ys,
                                         classifier=Core.classifier, tar_feature=Core.Xt)

    sum = abs(src_err) + abs(diff_marg) + abs(tar_err)
    if diff_marg == 0:
        FitnessFunction.margWeight = 0
    else:
        FitnessFunction.margWeight = diff_marg / sum

    if tar_err == 0:
        FitnessFunction.tarWeight = 0
    else:
        FitnessFunction.tarWeight = tar_err / sum

    if src_err == 0:
        FitnessFunction.srcWeight = 0
    else:
        FitnessFunction.srcWeight = src_err / sum


# args[0]: run_index
# args[1]: diff_ST: 1-domain classification, 2-MMD
# args[2]: tarErr: 1-Gecco, 2-pseudo with classification, 3-pseudo with silhouette,
#                   4-pseudo with MMDs,  5-pseudo with classification cycle
def main(args):
    global i_stick, i_pbest, i_gbest, ustks

    run_index = int(args[0])
    random.seed(1617 ** 2 * run_index)

    marg_index = int(args[1])
    tar_index = int(args[2])
    FitnessFunction.margVersion = marg_index
    FitnessFunction.tarVersion = tar_index
    FitnessFunction.srcVersion = 1

    filename = "iteration"+str(args[0])+".txt"
    output_file = open(filename, 'w+')

    time_start = time.clock()

    # Set the weight for each components in the fitness function
    # normalize_weight()
    FitnessFunction.srcWeight = 1.0
    FitnessFunction.margWeight = 0.0
    FitnessFunction.tarWeight = 1.0
    # print("Start setting weight")
    # FitnessFunction.set_weight(Core.src_feature, Core.src_label, Core.tar_feature)
    # print(FitnessFunction.srcWeight, FitnessFunction.margWeight, FitnessFunction.tarWeight)
    # print("End setting")
    # opposite_weight()

    # Initialize population and the gbest
    pop = toolbox.population(n=NPART)
    best = None

    to_write = ("Core classifier: %s\nSource weight: %f\nDiff source and target weight: %f\n"
                "Target weight: %g\nMarginal version: %d\nTarget version: %d\n"
                % (str(Core.classifier), FitnessFunction.srcWeight,
                   FitnessFunction.margWeight, FitnessFunction.tarWeight,
                   FitnessFunction.margVersion, FitnessFunction.tarVersion))

    archive = []

    for g in range(NGEN):
        print(g)
        to_write += ("=====Gen %d=====\n" % g)

        for part in pop:
            # Evaluate all particles
            part.fitness.values = toolbox.evaluate(part)

            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

            # update gbest
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        if TEST:
            print("is=", i_stick, "ip=", i_pbest, "ig=", i_gbest, "ustks=", ustks)
            print("best=", best)
            print(best.fitness.values)
            print("\n")
            for i, part in enumerate(pop):
                print("Particle %d: " % i)
                print("Particle position:", part)
                print("Particle pbest:", part.best)
                print("Particle stickiness:", part.stk)
                print("\n")
        archive.append(best)

        # now update the position of each particle
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitness components of the gbest and print the stats
        indices = [index for index, entry in enumerate(best) if entry == 1.0]
        src_feature = Core.Xs[:, indices]
        tar_feature = Core.Xt[:, indices]
        src_err, diff_marg, tar_err = \
            FitnessFunction.domain_differece(src_feature=src_feature, src_label=Core.Ys,
                                             classifier=Core.classifier, tar_feature=tar_feature)

        to_write += ("  Source Error: %f \n  Marginal Difference: %f \n  Target Error: %f \n"
                     % (src_err, diff_marg, tar_err))
        to_write += ("  Fitness function of real best: %f\n" % best.fitness.values[0])
        acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.Ys,
                                                         classifier=Core.classifier,
                                                         testing_feature=tar_feature, testing_label=Core.Yt)
        to_write += ("  Accuracy on unlabel target: " + str(acc) + "\n")
        to_write += "  Position:"+str(best)+"\n"
        print(src_err, acc, best.fitness.values[0])

        # update the parameters
        i_stick = is_up - (is_up - is_low)*(g+1)/NGEN
        i_gbest = (1-i_stick)/(pg_rate+1)
        i_pbest = pg_rate*i_gbest
        ustks = ustks_low + (ustks_up-ustks_low)*(g+1)/NGEN

    time_elapsed = (time.clock() - time_start)
    to_write += "----Final -----\n"
    indices = [index for index, entry in enumerate(best) if entry == 1.0]
    src_feature = Core.Xs[:, indices]
    tar_feature = Core.Xt[:, indices]

    if WRITE_OUT:
        src_full = np.concatenate((src_feature, np.reshape(Core.Ys, (Core.Ys.shape[0], 1))), axis=1)
        tar_full = np.concatenate((tar_feature, np.reshape(Core.Yt, (Core.Yt.shape[0], 1))), axis=1)
        np.savetxt("OutGP/Source", src_full, delimiter=",")
        np.savetxt("OutGP/Target", tar_full, delimiter=",")
        nf_file = open("OutGP/noFeatures", "w")
        nf_file.write(str(len(indices)))

    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.Ys,
                                                     classifier=Core.classifier,
                                                     testing_feature=tar_feature, testing_label=Core.Yt)
    to_write += ("Accuracy of the core classifier: " + str(acc) + "\n")
    to_write += ("Accuracy on the target (No TL) (core classifier): %f\n\n" % (
                    1.0 - FitnessFunction.classification_error(training_feature=Core.ori_src_feature,
                                                               training_label=Core.Ys,
                                                               classifier=Core.classifier,
                                                               testing_feature=Core.ori_tar_feature,
                                                               testing_label=Core.Yt)))
    new_classifier = LinearSVC(random_state=1617)
    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.Ys,
                                                     classifier=new_classifier,
                                                     testing_feature=tar_feature, testing_label=Core.Yt)
    to_write += ("Accuracy of the Linear SVM classifier: " + str(acc) + "\n")
    to_write += ("Accuracy on the target (No TL) of Linear SVM: %f\n\n" % (
            1.0 - FitnessFunction.classification_error(training_feature=Core.ori_src_feature,
                                                       training_label=Core.Ys,
                                                       classifier=new_classifier,
                                                       testing_feature=Core.ori_tar_feature,
                                                       testing_label=Core.Yt)))

    new_classifier = DecisionTreeClassifier(random_state=1617)
    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.Ys,
                                                     classifier=new_classifier,
                                                     testing_feature=tar_feature, testing_label=Core.Yt)
    to_write += ("Accuracy of the Linear DT classifier: " + str(acc) + "\n")
    to_write += ("Accuracy on the target (No TL) of DT: %f\n\n" % (
            1.0 - FitnessFunction.classification_error(training_feature=Core.ori_src_feature,
                                                       training_label=Core.Ys,
                                                       classifier=new_classifier,
                                                       testing_feature=Core.ori_tar_feature,
                                                       testing_label=Core.Yt)))

    new_classifier = GaussianNB()
    acc = 1.0 - FitnessFunction.classification_error(training_feature=src_feature, training_label=Core.Ys,
                                                     classifier=new_classifier,
                                                     testing_feature=tar_feature, testing_label=Core.Yt)
    to_write += ("Accuracy of the Linear NB classifier: " + str(acc) + "\n")
    to_write += ("Accuracy on the target (No TL) of NB: %f\n\n" % (
            1.0 - FitnessFunction.classification_error(training_feature=Core.ori_src_feature,
                                                       training_label=Core.Ys,
                                                       classifier=new_classifier,
                                                       testing_feature=Core.ori_tar_feature,
                                                       testing_label=Core.Yt)))

    to_write += ("Computation time: %f\n" % time_elapsed)
    to_write += ("Number of features: %d\n" % len(indices))
    to_write += str(best)

    output_file.write(to_write)
    output_file.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
