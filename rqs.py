import time
import os
import numpy as np
import csv
import statistics
import sys
from scipy.optimize import lsq_linear

from functions.metric import calcHV
from algorithm import nsga2
from algorithm import domination
from functions.function import fitnessFunctionTime_detectedMutants, generate_scores, loadTCDData, costFunction, constructSystem


def main(project, best_fitness):
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

    repeat = 20

    # open write file
    writePath = os.path.abspath("result/" + str(project) + "test.csv")

    with open(writePath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['nsga2_tet', 'nsga2_derivative', 'nsga2_infinite', 'nsga2_instability', 'nsga2_minmax',
                         'nsga2_ms', 'nsga2_hv', 'nsga2_time',
                         'less_tet', 'less_derivative', 'less_infinite', 'less_instability', 'less_minmax',
                         'less_ms', 'less_hv', 'less_time'])

        # define best fitness combination
        # best_fitness = ['time', 'discontinuity']
        all_fitness = ['time', 'discontinuity', 'infinite', 'instability', 'minmax']

        # repeat n times
        for i in range(repeat):
            print("repeat " + str(i))

            # initialize write row
            output_row = []

            ###### nsga2 starts here ######
            print("nsga2 starts")
            pop_size = 100
            max_gen = 250
            crossover_prob = 0.8
            mutate_prob = 1 / number_tc

            start_time_nsga2 = time.time()

            # create starting population
            population = np.array(nsga2.generate_pop(pop_size, number_tc))

            for generation in range(max_gen):
                population = nsga2.breed_population1(population, crossover_prob, mutate_prob)

                # score population
                scores = nsga2.score_population(population, time_metric, number_tc, project, best_fitness)

                # build pareto front
                population = nsga2.build_pareto_population(population, scores, pop_size)

            end_time_nsga2 = time.time()
            population_nsga2 = population

            nsga2_scores = nsga2.score_population(population_nsga2, time_metric, number_tc, project, best_fitness)
            population_nsga2_ids = np.arange(population_nsga2.shape[0]).astype(int)
            pareto_front_nsga2 = nsga2.identify_pareto(nsga2_scores, population_nsga2_ids)
            population_nsga2 = population_nsga2[pareto_front_nsga2, :]

            # calculate black box metrics
            nsga2_bb_scores = []
            for item in population_nsga2:
                nsga2_bb_scores.append(costFunction(item, time_metric, number_tc, project, [], [], [], []))

            for iteration in range(len(nsga2_bb_scores[0])):
                output_row.append(round(statistics.mean([nsga2_bb_scores[j][iteration] for j in range(len(nsga2_bb_scores))]), 2))

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
            ###### end nsga2 ######

            ###### less starts here ######
            print("less starts")
            pop_size = 10000

            sway_scores = generate_scores(pop_size)
            sumTime, sunDerivative, sumInfinite, sumInstability, sumMinmax, A = constructSystem(project, number_tc, time_metric)

            start_time_less = time.time()
            # new_res = sway.sway_runner(sway_scores)
            new_res = domination.domination_runner(sway_scores)

            new_scores = []
            for item in new_res:
                temp_score = item
                b = np.array([temp_score[0]*sumTime, temp_score[1]*sunDerivative, temp_score[2]*sumInfinite,
                              temp_score[3]*sumInstability, temp_score[4]*sumMinmax])

                x = lsq_linear(A, b, bounds=(0, 1))
                sol = x["x"]

                new_sol = []
                for ii in range(len(sol)):
                    if sol[ii] >= 0.4:
                        new_sol.append(1)
                    else:
                        new_sol.append(0)

                new_scores.append([new_sol, costFunction(new_sol, time_metric, number_tc, project, [], [], [], [])])

            end_time_less = time.time()

            population_less = np.array([new_scores[i][0] for i in range(len(new_scores))])
            less_scores = nsga2.score_population(population_less, time_metric, number_tc, project, all_fitness)
            population_less_ids = np.arange(population_less.shape[0]).astype(int)
            pareto_front_less = nsga2.identify_pareto(less_scores, population_less_ids)
            population_less = population_less[pareto_front_less, :]

            # calculate black box metrics
            less_bb_scores = []
            for item in population_less:
                less_bb_scores.append(costFunction(item, time_metric, number_tc, project, [], [], [], []))

            for iteration in range(len(less_bb_scores[0])):
                output_row.append(
                    round(statistics.mean([less_bb_scores[j][iteration] for j in range(len(less_bb_scores))]), 2))

            # calculate black box metrics
            # for iteration in range(len(new_scores[0][1])):
            #     temp_list = [new_scores[j][1][iteration] for j in range(len(new_scores))]
            #
            #     output_row.append(round(statistics.mean(temp_list), 2))

            # calculate evaluation metric - hv and average weighted sum of mutation score and normalized test execution time
            # temp_pop = [new_scores[j][0] for j in range(len(new_scores))]

            G2 = []
            tet2 = []
            mutantScore2 = []
            # for item in temp_pop:
            for item in population_less:
                t, m = fitnessFunctionTime_detectedMutants(item, time_metric, project)
                tet2.append(t)
                mutantScore2.append(m)
                G2.append([t, m])

            less_runtime = end_time_less - start_time_less
            less_hv = calcHV(G2)

            output_row.append(round(statistics.mean(mutantScore2), 2))
            output_row.append(round(less_hv, 2))
            output_row.append(round(less_runtime, 2))
            ###### end less ######

            writer.writerow(output_row)


if __name__ == "__main__":
    print("usage:")
    print("-p [project]: clean the results of that project")

    if len(sys.argv) <= 1:
        print("please specify one project")
    else:
        if "-p" in sys.argv and "-f" in sys.argv:
            project = sys.argv[sys.argv.index("-p") + 1]
            fitness = sys.argv[sys.argv.index("-f") + 1]

            best_fitness = []
            for item in fitness.split(","):
                best_fitness.append(item)

            main(project, best_fitness)
        elif "-p" not in sys.argv:
            print("please use -p command to enter the project name")
        elif "-f" not in sys.argv:
            print("please use -f to indicate the best combination of objectives")
