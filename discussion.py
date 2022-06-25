import time
import os
from scipy import io
from scipy.optimize import curve_fit
import csv
import numpy as np
import statistics
import random
import sys
import matplotlib.pyplot as plt
import math

from functions.function import allTestCases_Derivative, allTestCases_Instability, allTestCases_Infinite, \
    allTestCases_MinMax, fitnessFunctionTime_detectedMutants

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

def objective(x, a, b, c):
    return a * x + b * x**2 + c

def main(project):
    # set project and read TCD data
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

    all_fitness = ['time', 'discontinuity', 'infinite', 'instability', 'minmax']

    x_derivative = {}
    x_instability = {}
    x_infinite = {}
    x_minmax = {}

    for n_test in range(1, 100):
        print("cur in " + str(n_test))
        scanned = []

        c = 0
        while c < 30:
            cur_idx = sorted(random.sample(range(1, number_tc), n_test))

            if cur_idx not in scanned:
                scanned.append(cur_idx)
                c += 1

        for item in scanned:
            population = [0 for k in range(number_tc)]

            for ii in item:
                population[ii-1] = 1

            scores = [allTestCases_Derivative(project, population), allTestCases_Instability(project, population),
                      allTestCases_Infinite(project, population), allTestCases_MinMax(project, population)]

            t, m = fitnessFunctionTime_detectedMutants(population, time_metric, project)

            if scores[0] not in x_derivative.keys():
                x_derivative[scores[0]] = [1-m]
            else:
                x_derivative[scores[0]].append(1-m)

            if scores[1] not in x_instability.keys():
                x_instability[scores[1]] = [1-m]
            else:
                x_instability[scores[1]].append(1-m)

            if scores[2] not in x_infinite.keys():
                x_infinite[scores[2]] = [1-m]
            else:
                x_infinite[scores[2]].append(1-m)

            if scores[3] not in x_minmax.keys():
                x_minmax[scores[3]] = [1-m]
            else:
                x_minmax[scores[3]].append(1-m)

    for key in x_derivative.keys():
        x_derivative[key] = statistics.median(x_derivative[key])

    for key in x_instability.keys():
        x_instability[key] = statistics.median(x_instability[key])

    for key in x_infinite.keys():
        x_infinite[key] = statistics.median(x_infinite[key])

    for key in x_minmax.keys():
        x_minmax[key] = statistics.median(x_minmax[key])

    plt.scatter(list(x_derivative.keys()), [x_derivative[key] for key in x_derivative.keys()])
    popt, _ = curve_fit(objective, list(x_derivative.keys()), [x_derivative[key] for key in x_derivative.keys()])
    a, b, c = popt
    x_line = np.arange(min(list(x_derivative.keys())), max(list(x_derivative.keys())), 0.0001)
    y_line = objective(x_line, a, b, c)
    plt.plot(x_line, y_line, '--', c='k')
    plt.show()

    plt.scatter(list(x_instability.keys()), [x_instability[key] for key in x_instability.keys()])
    popt, _ = curve_fit(objective, list(x_instability.keys()), [x_instability[key] for key in x_instability.keys()])
    a, b, c = popt
    x_line = np.arange(min(list(x_instability.keys())), max(list(x_instability.keys())), 0.0001)
    y_line = objective(x_line, a, b, c)
    plt.plot(x_line, y_line, '--', c='k')
    plt.show()

    plt.scatter(list(x_infinite.keys()), [x_infinite[key] for key in x_infinite.keys()])
    popt, _ = curve_fit(objective, list(x_infinite.keys()), [x_infinite[key] for key in x_infinite.keys()])
    a, b, c = popt
    x_line = np.arange(min(list(x_infinite.keys())), max(list(x_infinite.keys())), 0.0001)
    y_line = objective(x_line, a, b, c)
    plt.plot(x_line, y_line, '--', c='k')
    plt.show()

    plt.scatter(list(x_minmax.keys()), [x_minmax[key] for key in x_minmax.keys()])
    popt, _ = curve_fit(objective, list(x_minmax.keys()), [x_minmax[key] for key in x_minmax.keys()])
    a, b, c = popt
    x_line = np.arange(min(list(x_minmax.keys())), max(list(x_minmax.keys())), 0.0001)
    y_line = objective(x_line, a, b, c)
    plt.plot(x_line, y_line, '--', c='k')
    plt.show()


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