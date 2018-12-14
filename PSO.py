#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from deap import base
from deap import creator
from deap import tools
import FitnessFunction
import Core

# Setting from Problem
NBIT = (len(Core.Xs) + len(Core.Xt)) * Core.C
NGEN = 1000
NPART = 100#NBIT if NBIT < 100 else 100

# PSO parameters
w = 0.7298
c1 = 1.49618
c2 = 1.49618
pos_max = 10.0
pos_min = -10.0
s_max = (pos_max - pos_min)/30
s_min = -s_max

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin,
               speed=list, smin=None, smax=None, best=None,
               target_predict=list)


def generate(size, pmin, pmax, smin, smax):
    # create position
    position = np.random.uniform(pmin, pmax, size)

    #refine the position
    beta = np.copy(position)
    beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
    beta = FitnessFunction.refine(beta)
    position = np.reshape(beta, ((len(Core.Xs) + len(Core.Xt))*Core.C),)

    part = creator.Particle(position)
    # create speed
    part.speed = np.zeros(size)
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best):
    # calculate speed
    u1 = np.random.uniform(0, 1, len(part))
    u2 = np.random.uniform(0, 1, len(part))
    v_u1 = c1 * u1 * (part.best - part)
    v_u2 = c2 * u2 * (best - part)
    part.speed = w * part.speed + v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax

    # calculate new positions
    part += part.speed
    for i, entry in enumerate(part):
        if entry < pos_min:
            part[i] = pos_min
        elif entry > pos_max:
            part[i] = pos_max


def evaluate(particle):
    beta = np.copy(particle)
    beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
    fitness = FitnessFunction.fitness_function(beta)
    return fitness,


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=NBIT, pmin=pos_min, pmax=pos_max, smin=s_min, smax=s_max)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle)
toolbox.register("evaluate", evaluate)


def main(args):
    run_index = int(args[0])
    np.random.seed(1617 ** 2 * run_index)

    pop = toolbox.population(n=NPART)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
    best = None
    glive = 0
    gbest_try = False

    for g in range(NGEN):
        print('==============Gen %d===============' %g)

        gbest_update = False
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
                part.best.target_predict = part.target_predict
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
                best.target_predict = part.target_predict
                gbest_update = True

        if gbest_update:
            glive = 1
            gbest_try = False
        else:
            glive = glive + 1

        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

        if glive > 10:
            print('Start reparing gbest')

            if gbest_try:
                beta = np.random.uniform(pos_min, pos_max, NBIT)
            else:
                beta = np.copy(best)
                gbest_try = True

            beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
            beta = FitnessFunction.refine(beta)
            position = np.reshape(beta, ((len(Core.Xs) + len(Core.Xt)) * Core.C), )
            tmp = creator.Particle(position)
            tmp.fitness.values = toolbox.evaluate(tmp)
            if best.fitness < tmp.fitness:
                print("Update gbest")
                best = tmp
                glive = 1
                gbest_try = False

        # update the target pseudo
        beta = np.copy(best)
        beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
        F = np.dot(Core.K, beta)
        Cls = np.argmax(F, axis=1) + 1

        label_target = Cls[Core.ns:]
        Core.Yt_pseu = label_target
        acc_target = np.mean(label_target == Core.Yt)

        label_source = Cls[:Core.ns]
        acc_source = np.mean(label_source == Core.Ys)

        print("Source acc: %f Target acc: %f" % (acc_source, acc_target))
        print(best.fitness)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])