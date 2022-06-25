import time
import os
from scipy import io
import csv
import numpy as np
import statistics
import sys

from algorithm.nsga3 import nsga_iii_runner
from functions.function import costFunction, fitnessFunctionTime_detectedMutants
from algorithm import nsga2
from functions.metric import calcHV


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
    # project = "Twotanks"

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
    writePath = os.path.abspath("result/" + str(project) + "_nsga3.csv")
    with open(writePath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['nsga3_tet', 'nsga3_derivative', 'nsga3_infinite', 'nsga3_instability',
                         'nsga3_minmax', 'nsga3_ms', 'nsga3_hv', 'nsga3_time'])

        for i in range(repeat):
            output_row = []
            print("repeat ", i+1)

            # start nsga-iii here
            start_time = time.time()

            population, log_book = nsga_iii_runner(crossover_prob, max_gen, pop_size, time_metric, number_tc, project)

            end_time = time.time()

            population_nsga3 = np.array(population)
            nsga3_scores = nsga2.score_population(population_nsga3, time_metric, number_tc, project, all_fitness)
            population_nsga3_ids = np.arange(population_nsga3.shape[0]).astype(int)
            pareto_front_nsga3 = nsga2.identify_pareto(nsga3_scores, population_nsga3_ids)
            population_nsga3 = population_nsga3[pareto_front_nsga3, :]

            # calculate black box metrics
            nsga3_bb_scores = []
            for item in population_nsga3:
                nsga3_bb_scores.append(costFunction(item, time_metric, number_tc, project, [], [], [], []))

            for iteration in range(len(nsga3_bb_scores[0])):
                output_row.append(
                    round(statistics.mean([nsga3_bb_scores[j][iteration] for j in range(len(nsga3_bb_scores))]), 2))

            G = []
            tet = []
            mutantScore = []
            # for item in temp_pop:
            for item in population_nsga3:
                t, m = fitnessFunctionTime_detectedMutants(item, time_metric, project)
                tet.append(t)
                mutantScore.append(m)
                G.append([t, m])

            nsga3_runtime = end_time - start_time
            nsga3_hv = calcHV(G)

            output_row.append(round(statistics.mean(mutantScore), 2))
            output_row.append(round(nsga3_hv, 2))
            output_row.append(round(nsga3_runtime, 2))

            writer.writerow(output_row)


if __name__ == "__main__":
    print("usage:")
    print("-p [project]: clean the results of that project")

    if len(sys.argv) <= 1:
        print("please specify one project")
    else:
        if "-p" in sys.argv:
            project = sys.argv[sys.argv.index("-p")+1]
            main(project)
        else:
            print("please use -p command to enter the project name")




