from algorithm import nsga2
from functions.metric import calcHV
from functions.function import costFunction, fitnessFunctionTime_detectedMutants
from model.csp import CSP

import numpy as np
import random
from scipy import io
import os
import statistics
import time
import csv
import sys

from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import BinaryRandomSampling
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation

def loadTCDData(project):
    if project == "Twotanks":
        path = "data/" + str(project) + "/TCData.mat"
        filePath = os.path.abspath(path)
    elif project == "ACEngine":
        path = "data/" + str(project) + "/FDC_DATA.mat"
        filePath = os.path.abspath(path)
    elif project == "EMB":
        path = "data/" + str(project) + "/BlackBoxMetrics_2.mat"
        filePath = os.path.abspath(path)
    elif project == "CW":
        path = "data/" + str(project) + "/TC_time.mat"
        filePath = os.path.abspath(path)
    elif project == "CC":
        path = "data/" + str(project) + "/TCData.mat"
        filePath = os.path.abspath(path)
    elif project == "Tiny":
        path = "data/" + str(project) + "/TCData.mat"
        filePath = os.path.abspath(path)

    return io.loadmat(filePath)

def main(project):
    # set project and read TCD data
    # project = "CW"

    # set param for each project
    if project == "Twotanks":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 11, 7
        number_tc = 150
        time_metric = [TCD['TCData'][0][i][0][0][0] for i in range(number_tc)]
    elif project == "ACEngine":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 4, 1
        number_tc = 120
        time_metric = [TCD['test_case'][0][i][1][0][0] for i in range(number_tc)]
    elif project == "EMB":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 1, 1
        number_tc = 150
        time_metric = [TCD['TCData'][0][i][0][0][-1][0][0] for i in range(number_tc)]
    elif project == "CW":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 15, 4
        number_tc = 133
        time_metric = [TCD['time_testCases'][i][0] for i in range(number_tc)]
    elif project == "CC":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 5, 2
        number_tc = 150
        time_metric =[TCD['TCData'][0][i][0][0][-1][0][0] for i in range(number_tc)]
    elif project == "Tiny":
        TCD = loadTCDData(project)
        nInputs, nOutputs = 3, 1
        number_tc = 150
        time_metric = [TCD['TCData'][0][i][0][0][-1][0][0] for i in range(number_tc)]

    # define parameters
    repeat = 20
    pop_size = 100
    max_gen = 250
    crossover_prob = 0.8
    mutate_prob = 1 / number_tc
    all_fitness = ['time', 'discontinuity', 'infinite', 'instability', 'minmax']

    # open write file
    writePath = os.path.abspath("result/" + str(project) + "_moead.csv")
    with open(writePath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['moead_tet', 'moead_derivative', 'moead_infinite', 'moead_instability',
                         'moead_minmax', 'moead_ms', 'moead_hv', 'moead_time'])

        for i in range(repeat):
            output_row = []
            print("repeat ", i + 1)

            # start moea/d here
            start_time = time.time()

            problem = CSP(project, time_metric, number_tc)

            algorithm = MOEAD(get_reference_directions("das-dennis", 3, n_partitions=12),
                              n_neighbors=15,
                              decomposition="pbi",
                              prob_neighbor_mating=0.9,
                              seed=random.random(),
                              sampling=BinaryRandomSampling(),
                              crossover=SimulatedBinaryCrossover(prob=crossover_prob, eta=20),
                              mutation=BinaryBitflipMutation(prob=mutate_prob))

            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', max_gen),
                           verbose=False)

            end_time = time.time()

            population_moead = np.array(res.pop.get("X"))
            moead_scores = nsga2.score_population(population_moead, time_metric, number_tc, project, all_fitness)
            population_moead_ids = np.arange(population_moead.shape[0]).astype(int)
            pareto_front_moead = nsga2.identify_pareto(moead_scores, population_moead_ids)
            population_moead = population_moead[pareto_front_moead, :]

            # calculate black box metrics
            moead_bb_scores = []
            for item in population_moead:
                moead_bb_scores.append(costFunction(item, time_metric, number_tc, project, [], [], [], []))

            for iteration in range(len(moead_bb_scores[0])):
                output_row.append(
                    round(statistics.mean([moead_bb_scores[j][iteration] for j in range(len(moead_bb_scores))]), 2)
                )

            G = []
            tet = []
            mutantScore = []
            # for item in temp_pop:
            for item in population_moead:
                t, m = fitnessFunctionTime_detectedMutants(item, time_metric, project)
                tet.append(t)
                mutantScore.append(m)
                G.append([t, m])

            moead_runtime = end_time - start_time
            moead_hv = calcHV(G)
            output_row.append(round(statistics.mean(mutantScore), 2))
            output_row.append(round(moead_hv, 2))
            output_row.append(round(moead_runtime, 2))

            writer.writerow(output_row)


if __name__ == "__main__":
    print("usage:")
    print("-p [project]: clean the results of that project")

    if len(sys.argv) <= 1:
        print("please specify one project")
    else:
        if "-p" in sys.argv:
            project = sys.argv[sys.argv.index("-p") + 1]
            main(project)
        else:
            print("please use -p command to enter the project name")