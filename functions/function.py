from scipy import io
import random
import pandas as pd
import numpy as np
import os

def loadMatData(file):
    return io.loadmat(file)

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

def time_func(time_metric, activationArray, number_tc):
    time = 0
    total_time = 0

    for i in range(number_tc):
        total_time += time_metric[i]

        if activationArray[i] == 1:
            time += time_metric[i]

    return time / total_time

def allTestCases_Derivative(project, activationArray):
    filePath = os.path.abspath("data/" + str(project) + "/derivative.mat")
    derivative = loadMatData(filePath)['derivative']

    newDerivative, allDerivative = 0, 0

    for k in range(len(activationArray)):
        allDerivative += derivative[k][0]

        if activationArray[k] == 1:
            newDerivative += derivative[k][0]
    
    return newDerivative/allDerivative

def allTestCases_Infinite(project, activationArray):
    filePath = os.path.abspath("data/" + str(project) + "/infinite.mat")
    infinite = loadMatData(filePath)['infinite']

    newInf, allInf = 0, 0

    for k in range(len(activationArray)):
        allInf += infinite[k][0]

        if activationArray[k] == 1:
            newInf += infinite[k][0]
    
    return newInf/allInf

def allTestCases_Instability(project, activationArray):
    filePath = os.path.abspath("data/" + str(project) + "/instability.mat")
    instability = loadMatData(filePath)['instability']

    newInstability, allInstability = 0, 0

    for k in range(len(activationArray)):
        allInstability += instability[k][0]

        if activationArray[k] == 1:
            newInstability += instability[k][0]
    
    return newInstability/allInstability

def allTestCases_MinMax(project, activationArray):
    filePath = os.path.abspath("data/" + str(project) + "/minMax.mat")
    minmax = loadMatData(filePath)['minMax']

    newMinMax, allMinMax = 0, 0

    for k in range(len(activationArray)):
        allMinMax += minmax[k][0]

        if activationArray[k] == 1:
            newMinMax += minmax[k][0]

    return newMinMax/allMinMax

def calculate_Euclidean(activationArray, euclideanTable):
    NTC = 0
    outEuclidean = 0

    for i in range(len(activationArray)):
        if activationArray[i] != 0:
            NTC += 1
    
    if NTC > 1:
        for i in range(len(activationArray)):
            if activationArray[i] == 1:
                for j in range(len(activationArray)):
                    if activationArray[j] == 1:
                        outEuclidean += abs(euclideanTable[i][i] - euclideanTable[i][j])

        outEuclidean = outEuclidean / (len(activationArray) * ((len(activationArray) - 1) / 2))
    else:
        outEuclidean = 0
    
    return outEuclidean

def totalInputEuclidean(activationArray, inputE, nInputs):
    outInputEuclidean = []
    for i in range(nInputs):
        outInputEuclidean.append(calculate_Euclidean(activationArray, inputE[i]))
    
    a = sum(outInputEuclidean)

    return a / len(outInputEuclidean)

def totalOutputEuclidean(activationArray, outputE, nOutputs):
    outOutputEuclidean = []
    for i in range(nOutputs):
        outOutputEuclidean.append(calculate_Euclidean(activationArray, outputE[i]))
    
    a = sum(outOutputEuclidean)

    return a / len(outOutputEuclidean)

def costFunction(x, time_metric, number_tc, project, inputE, nInputs, outputE, nOutputs):
    time = 0
    totalTime = 0

    for i in range(number_tc):
        totalTime += time_metric[i]

        if x[i] == 1:
            time += time_metric[i]
    
    cost = time/totalTime
    discontinuity = allTestCases_Derivative(project, x)
    infinity = allTestCases_Infinite(project, x)
    instability = allTestCases_Instability(project, x)
    minmax = allTestCases_MinMax(project, x)
    # indistance = totalInputEuclidean(x, inputE, nInputs)
    # outdistance = totalOutputEuclidean(x, outputE, nOutputs)

    # return [cost, 1-discontinuity, 1-infinity, 1-instability, 1-minmax, 1-indistance, 1-outdistance]
    return [cost, discontinuity, infinity, instability, minmax]

def generate_table(time_metric, project, nInputs, nOutputs, number_tc):
    # collect input data
    inputE = []
    for i in range(nInputs):
        inputFilePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/inputEuclidean_" + str(i+1) + ".mat"
        inputE.append(loadMatData(inputFilePath)['inputEuclidean'])

    # collect output data
    outputE = []
    for i in range(nOutputs):
        outputFilePath = "/mnt/e/Research/IST2019Paper/IST2019PY/data/" + str(project) + "/outputEuclidean_" + str(i+1) + ".mat"
        outputE.append(loadMatData(outputFilePath)['OutputEuclidean'])

    # define parameters  # need to change to 10000
    nPop = 1000

    # generate population table
    popTable = []
    for p in range(nPop):
        # if p%100 == 0:
        #     print("generate ", p)
        tempArray = [random.randint(0, 1) for _ in range(number_tc)]
        cost = costFunction(tempArray, time_metric, number_tc, project, inputE, nInputs, outputE, nOutputs)
        popTable.append((tempArray, cost))
        
    return popTable

def detectedMutants(activationArray, project):
    filePath = os.path.abspath("data/" + str(project) + "/MutantsMatrixTable.xlsx")
    mutantsTable = pd.read_excel(filePath, engine='openpyxl').to_numpy()

    detectedMutantsArray = []

    k = 1
    a, b = len(mutantsTable), len(mutantsTable[0])

    for i in range(len(activationArray)):
        if activationArray[i] == 1:
            for j in range(b):
                if mutantsTable[i][j] == 1:
                    detectedMutantsArray.append(j)

    detectedMutantsArray = np.unique(np.array(detectedMutantsArray))

    return len(detectedMutantsArray)/b

def fitnessFunctionTime_detectedMutants(activationArray, time_metric, project):
    popSize = len(activationArray)

    time = 0
    totalTime = 0

    for i in range(len(time_metric)):
        totalTime += time_metric[i]

        if activationArray[i] == 1:
            time += time_metric[i]
    
    detectedMutantsP = detectedMutants(activationArray, project)

    return time/totalTime, 1 - detectedMutantsP

def generate_scores(pop_size):
    population = []
    
    for i in range(pop_size):
        population.append([[], [random.uniform(0.15, 1) for _ in range(5)]])
    
    return population


def constructSystem(project, number_tc, time_metric):
    derivativeFilePath = os.path.abspath("data/" + str(project) + "/derivative.mat")
    derivative = loadMatData(derivativeFilePath)['derivative']

    infiniteFilePath = os.path.abspath("data/" + str(project) + "/infinite.mat")
    infinite = loadMatData(infiniteFilePath)['infinite']

    instabilityFilePath = os.path.abspath("data/" + str(project) + "/instability.mat")
    instability = loadMatData(instabilityFilePath)['instability']

    minmaxFilePath = os.path.abspath("data/" + str(project) + "/minMax.mat")
    minmax = loadMatData(minmaxFilePath)['minMax']

    # initilize
    sumTime, sumDerivative, sumInfinite, sumInstability, sumMinmax = 0, 0, 0, 0, 0
    time_list, derivative_list, infinite_list, instability_list, minmax_list = [], [], [], [], []

    for i in range(number_tc):
        sumTime += time_metric[i]
        sumDerivative += derivative[i][0]
        sumInfinite += infinite[i][0]
        sumInstability += instability[i][0]
        sumMinmax += minmax[i][0]

        time_list.append(time_metric[i])
        derivative_list.append(derivative[i][0])
        infinite_list.append(infinite[i][0])
        instability_list.append(instability[i][0])
        minmax_list.append(minmax[i][0])

    result_list = [time_list, derivative_list, infinite_list, instability_list, minmax_list]

    return sumTime, sumDerivative, sumInfinite, sumInstability, sumMinmax, np.array(result_list)

def main():
    t = generate_table()

if __name__ == "__main__":
    main()