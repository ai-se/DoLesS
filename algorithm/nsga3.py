# ref: the implementation of nsga3 algorithm is based on DEAP: https://deap.readthedocs.io/en/master/index.html

import copy
import random
import numpy as np
import statistics
from deap import algorithms, base, tools, creator
from functions.function import costFunction

# NSGA-III DEAP implementation starts here
# prepare for toolbox
def prepare_toolbox(problem_instance, crossover_prob, max_gen, pop_size,
                    time_metric, number_tc, project):
    def binary(n):
        return [random.randint(0, 1) for _ in range(n)]

    toolbox = base.Toolbox()

    toolbox.register("attribute", binary, n=number_tc)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / number_tc)

    # toolbox.register('select', selection_func,
    #                  time_metric=time_metric, number_tc=number_tc, project=project)
    ref_points = tools.uniform_reference_points(nobj=5, p=20)
    toolbox.register('select', tools.selNSGA3, ref_points=ref_points)

    toolbox.register('evaluate', problem_instance, time_metric=time_metric, number_tc=number_tc, project=project)

    toolbox.pop_size = pop_size  # population size
    toolbox.max_gen = max_gen  # max number of iteration
    toolbox.mut_prob = 1 / number_tc
    toolbox.cross_prob = crossover_prob

    return toolbox

# define evaluation function
def evaluate(individual, time_metric, number_tc, project):
    return costFunction(individual, time_metric, number_tc, project, [], [], [], [])

# nsga-iii algorithm entry
def nsga_iii(toolbox, stats=None, verbose=True):
    population = toolbox.population(n=toolbox.pop_size)
    return algorithms.eaMuPlusLambda(population, toolbox,
                              mu=toolbox.pop_size,
                              lambda_=toolbox.pop_size,
                              cxpb=toolbox.cross_prob,
                              mutpb=toolbox.mut_prob,
                              ngen=toolbox.max_gen,
                              stats=stats, verbose=verbose)

# runner for nsga-iii
def nsga_iii_runner(crossover_prob, max_gen, pop_size, time_metric, number_tc, project):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = prepare_toolbox(evaluate, crossover_prob, max_gen, pop_size, time_metric, number_tc, project)

    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    res, logbook = nsga_iii(toolbox, stats=stats)

    return res, logbook

# __all__ = ["sel_nsga_iii"]