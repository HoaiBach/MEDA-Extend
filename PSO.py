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
from sklearn.metrics import accuracy_score as score
import FitnessFunction
import Core

# Setting from Problem
NBIT = (len(Core.Xs) + len(Core.Xt)) * Core.C
NGEN = 100
NPART = 30#NBIT if NBIT < 100 else 100

# PSO parameters
w = 0.7298
c1 = 1.49618
c2 = 1.49618
pos_max = 10
pos_min = -10
s_max = (pos_max - pos_min)/5
s_min = -s_max

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin,
               speed=list, smin=None, smax=None, best=None,
               target_predict=list)


def generate(size, pmin, pmax, smin, smax):
    position = np.random.uniform(pmin, pmax, size)
    part = creator.Particle(position)
    part.speed = np.zeros(size)
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best):
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
    part += part.speed
    for i, entry in enumerate(part):
        if entry < pos_min:
            part[i] = pos_min
        elif entry > pos_max:
            part[i] = pos_max


def evaluate(particle):
    beta = np.copy(particle)
    beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
    return FitnessFunction.fitness_function(beta),


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

    for g in range(NGEN):
        print('==============Gen %d===============' %g)
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
        for part in pop:
            toolbox.update(part, best)

        # update the target pseudo
        beta = np.copy(best)
        beta = np.reshape(beta, (len(Core.Xs) + len(Core.Xt), Core.C))
        F = np.dot(Core.K, beta)
        Cls = np.argmax(F, axis=1) + 1
        Cls = Cls[Core.ns:]
        Core.Yt_pseu = Cls
        acc = np.mean(Core.Yt_pseu == Core.Yt)
        print(acc)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])