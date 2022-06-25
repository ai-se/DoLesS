from algorithm import nsga2
from functions import function, metric

import time
import csv
import numpy as np
import statistics
from scipy import io
import os
import sys

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
        time_metric = [TCD['TCData'][0][i][0][0][-1][0][0] for i in range(number_tc)]
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

    # collect input/output data
    # inputE = []
    # for i in range(nInputs):
    #     inputFilePath = os.path.abspath("data/" + str(project) + "/inputEuclidean_" + str(i+1) + ".mat")
    #     inputE.append(function.loadMatData(inputFilePath)['inputEuclidean'])
    #
    # collect output data
    # outputE = []
    # for i in range(nOutputs):
    #     outputFilePath = os.path.abspath("data/" + str(project) + "/outputEuclidean_" + str(i+1) + ".mat")
    #     outputE.append(function.loadMatData(outputFilePath)['OutputEuclidean'])

    # define combinations of fitness function
    # fitness_function = [['time', 'discontinuity'], ['time', 'instability'], ['time', 'infinite'], ['time', 'minmax'], ['time', 'input'], ['time', 'output'],
    #                         ['time', 'discontinuity', 'instability'], ['time', 'discontinuity', 'infinite'], ['time', 'discontinuity', 'minmax'],
    #                         ['time', 'discontinuity', 'input'], ['time', 'discontinuity', 'output'], ['time', 'instability', 'infinite'], ['time', 'instability', 'minmax'],
    #                         ['time', 'instability', 'input'], ['time', 'instability', 'output'], ['time', 'infinite', 'minmax'],  ['time', 'infinite', 'input'],
    #                         ['time', 'infinite', 'output'], ['time', 'minmax', 'input'], ['time', 'minmax', 'output'], ['time', 'input', 'output']]

    fitness_function = [['time', 'discontinuity'], ['time', 'instability'], ['time', 'infinite'], ['time', 'minmax'],
                        ['time', 'discontinuity', 'instability'], ['time', 'discontinuity', 'infinite'], ['time', 'discontinuity', 'minmax'],
                        ['time', 'instability', 'infinite'], ['time', 'instability', 'minmax'], ['time', 'infinite', 'minmax'],
                        ['time', 'discontinuity', 'instability', 'infinite', 'minmax']]

    # open write file
    writePath = os.path.abspath("result/" + str(project) + "_nsga2Select.csv")
    with open(writePath, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["fitness", "metric", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", 'time'])

        for fitness in fitness_function:
            print(fitness)
            tet = []
            ms = []
            hv = []
            runtime = []

            for r in range(repeat):
                print("current repeat " + str(r))

                start_time = time.time()
                population = np.array(nsga2.generate_pop(pop_size, number_tc))

                for generation in range(max_gen):
                    if generation % 50 == 0:
                        print("generation ", generation)

                    # print("building population")
                    population = nsga2.breed_population1(population, crossover_prob, mutate_prob)

                    # score population
                    # print("scoring population")
                    scores = nsga2.score_population(population, time_metric, number_tc, project, fitness)

                    # build pareto front
                    # print("building pareto front")
                    population = nsga2.build_pareto_population(population, scores, pop_size)

                duration = time.time() - start_time
                population_nsga2 = population

                # calculate evaluation metric
                G = []
                TET = []
                MS = []

                for item in population_nsga2:
                    t, m = function.fitnessFunctionTime_detectedMutants(item, time_metric, project)
                    TET.append(t)
                    MS.append(1-m)
                    G.append([t, m])
                
                temp_hv = metric.calcHV(G)
                mean_tet = statistics.mean(TET)
                mean_ms = statistics.mean(MS)

                tet.append(mean_tet)
                ms.append(mean_ms)
                hv.append(temp_hv)
                runtime.append(duration)

            # write result to csv
            temp_fitness = ""
            for item in fitness:
                if temp_fitness == "":
                    temp_fitness = temp_fitness + item
                else:
                    temp_fitness = temp_fitness + "&" + item
            
            csv_writer.writerow([temp_fitness, "tet"] + tet + [sum(runtime)])
            csv_writer.writerow([temp_fitness, "ms"] + ms + [sum(runtime)])
            csv_writer.writerow([temp_fitness, "hv"] + hv + [sum(runtime)])


if __name__ == '__main__':
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