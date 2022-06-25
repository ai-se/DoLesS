import time
import os
from scipy import io
import csv
import numpy as np
import statistics
import sys

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
    all_fitness = ['time', 'discontinuity', 'infinite', 'instability', 'minmax']

    # open write file
    writePath = os.path.abspath("result/" + str(project) + "_nsga2_5goals.csv")
    with open(writePath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['nsga2_5g_tet', 'nsga2_5g_derivative', 'nsga2_5g_infinite', 'nsga2_5g_instability',
                         'nsga2_5g_minmax', 'nsga2_5g_ms', 'nsga2_5g_hv', 'nsga2_5g_time'])

        for i in range(repeat):
            print("repeat " + str(i))

            # initialize write row
            output_row = []

            pop_size = 100
            max_gen = 250
            crossover_prob = 0.8
            mutate_prob = 1 / number_tc

            start_time_nsga2 = time.time()

            # create starting population
            population = np.array(nsga2.generate_pop(pop_size, number_tc))

            for generation in range(max_gen):
                # print("generation" + str(generation))
                # print("start breed")
                population = nsga2.breed_population1(population, crossover_prob, mutate_prob)

                # score population
                # print("start score")
                scores = nsga2.score_population(population, time_metric, number_tc, project, all_fitness)

                # build pareto front
                # print("start pareto")
                population = nsga2.build_pareto_population(population, scores, pop_size)

            end_time_nsga2 = time.time()
            population_nsga2 = population

            nsga2_scores = nsga2.score_population(population_nsga2, time_metric, number_tc, project, all_fitness)
            population_nsga2_ids = np.arange(population_nsga2.shape[0]).astype(int)
            pareto_front_nsga2 = nsga2.identify_pareto(nsga2_scores, population_nsga2_ids)
            population_nsga2 = population_nsga2[pareto_front_nsga2, :]

            # calculate black box metrics
            nsga2_bb_scores = []
            for item in population_nsga2:
                nsga2_bb_scores.append(costFunction(item, time_metric, number_tc, project, [], [], [], []))

            for iteration in range(len(nsga2_bb_scores[0])):
                output_row.append(
                    round(statistics.mean([nsga2_bb_scores[j][iteration] for j in range(len(nsga2_bb_scores))]), 2))

            # calculate evaluation metric - hv and average weighted sum of mutation score and normalized test execution time
            G1 = []
            tet1 = []
            mutantScore1 = []
            for item in population_nsga2:
                t, m = fitnessFunctionTime_detectedMutants(item, time_metric, project)
                tet1.append(t)
                mutantScore1.append(m)
                G1.append([t, m])

            nsga2_runtime = end_time_nsga2 - start_time_nsga2
            nsga2_hv = calcHV(G1)

            output_row.append(round(statistics.mean(mutantScore1), 2))
            output_row.append(round(nsga2_hv, 2))
            output_row.append(round(nsga2_runtime, 2))

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




