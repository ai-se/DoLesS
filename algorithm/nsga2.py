import random
import numpy as np
from functions.function import loadTCDData, time_func, allTestCases_Instability, allTestCases_Derivative, allTestCases_Infinite, allTestCases_MinMax
from functions.metric import is_pareto_efficient
from scipy import io

# ref: the nsga2 algorithm is from
# https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/

def generate_pop(pop_size, number_tc):
    pop = []

    for i in range(pop_size):
        pop.append([random.randint(0, 1) for _ in range(number_tc)])
    
    return pop

def score_population(population, time_metric, number_tc, project, fitness):
    scores = []

    for item in population:
        temp_fitness = []

        if 'time' in fitness:
            temp_fitness.append(time_func(time_metric, item, number_tc))
        if 'discontinuity' in fitness:
            temp_fitness.append(1-allTestCases_Derivative(project, item))
        if 'instability' in fitness:
            temp_fitness.append(1-allTestCases_Instability(project, item))
        if 'infinite' in fitness:
            temp_fitness.append(1-allTestCases_Infinite(project, item))
        if 'minmax' in fitness:
            temp_fitness.append(1-allTestCases_MinMax(project, item))
        # if 'input' in fitness:
        #     temp_fitness.append(1-totalInputEuclidean(item, inputE, nInputs))
        # if 'output' in fitness:
        #     temp_fitness.append(1-totalOutputEuclidean(item, outputE, nOutputs))

        scores.append(temp_fitness)
    
    return np.array(scores)

def calculate_crowding(scores):
    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalize scores
    # not sure if this part necessary

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        crowding[0] = 1
        crowding[population_size-1] = 1

        sorted_scores = np.sort(scores[:, col])
        sorted_scores_index = np.argsort(scores[:, col])

        # calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to original order
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # record crowding distances
        crowding_matrix[:, col] = sorted_crowding
    
    # sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

def reduce_by_crowding(scores, number_to_select):
    population_ids = np.arange(scores.shape[0])
    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))
    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):
        population_size = population_ids.shape[0]

        fighter1ID = random.randint(0, population_size - 1)
        fighter2ID = random.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID), axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)
            crowding_distances = np.delete(crowding_distances, (fighter1ID), axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]
            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)
            crowding_distances = np.delete(crowding_distances, (fighter2ID), axis=0)

    # Convert to integer 
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)
    
    return (picked_population_ids)

def breed_by_crossover(parent1, parent2):
    length = min(len(parent1), len(parent2))

    crossover_point = random.randint(1, length - 1)

    # create children
    child1 = np.hstack((parent1[0:crossover_point], parent2[crossover_point:]))
    child2 = np.hstack((parent2[0:crossover_point], parent1[crossover_point:]))

    return child1, child2

def breed_by_mutation(ind, mutate_prob):
    new_ind = []

    for i in range(len(ind)):
        if random.random() < mutate_prob:
            new_ind.append(type(ind[i])(not ind[i]))
        else:
            new_ind.append(ind[i])

    return new_ind

def breed_population(population):
    new_population = []
    population_size = population.shape[0]

    for i in range(int(population_size/2)):
        parent_1 = population[random.randint(0, population_size - 1)]
        parent_2 = population[random.randint(0, population_size - 1)]

        child_1, child_2 = breed_by_crossover(parent_1, parent_2)

        new_population.append(child_1)
        new_population.append(child_2)
    
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

    return population

def breed_population1(population, crossover_prob, mutate_prob):
    new_population = []

    for ind1, ind2 in zip(population[::2], population[1::2]):
        # perform one point crossover with probability 0.8
        if random.random() <= crossover_prob:
            cross_ind1, cross_ind2 = breed_by_crossover(ind1, ind2)
        else:
            cross_ind1 = ind1
            cross_ind2 = ind2

        # perform bit flip mutation with mutate probability 1/N
        mutate_ind1 = breed_by_mutation(cross_ind1, mutate_prob)
        mutate_ind2 = breed_by_mutation(cross_ind2, mutate_prob)

        new_population.append(mutate_ind1)
        new_population.append(mutate_ind2)

    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

    return population

def identify_pareto(scores, population_ids):
    population_size = scores.shape[0]

    pareto_front = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        for j in range(population_size):
            # check if j dominate i, yes then pareto_front[i] = 0
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                pareto_front[i] = 0
                # stop comparison for i
                break
    
    return population_ids[pareto_front]

def build_pareto_population(population, scores, population_size):
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])

    pareto_front = []

    while len(pareto_front) < population_size:
        temp_pareto_front = identify_pareto(scores[unselected_population_ids, :], unselected_population_ids)

        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)

        # do crowding distance selection if pareto size exceeds
        if combined_pareto_size > population_size:
            number_to_select = combined_pareto_size - population_size
            selected_individuals = (reduce_by_crowding(scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]
        
        # add latest pareto front to full pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))
    
    population = population[pareto_front.astype(int)]
    return population

def algo(population, max_gen, time_metric, number_tc, project, pop_size):
    # nsga ii algorithm
    for generation in range(max_gen):
        if generation % 10 == 0:
            print("generation ", generation)

        population = breed_population(population)

        # score population
        scores = score_population(population, time_metric, number_tc, project)

        # build pareto front
        population = build_pareto_population(population, scores, pop_size)
    
    return population

def main():
    # set project and read TCD data
    project = "CW"

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

    max_gen = 250
    fitness = ['time', 'discontinuity', 'instability']
    # fitness = ['time', 'discontinuity']
    # problem = define_problem(time_metric, number_tc, project, fitness)
    # res = nsga2_executor(problem, number_tc, max_gen)
    # print(res.G)
    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(res.F, color="red")
    # plot.show()

def main1():
    # set project and read TCD data
    project = "Twotanks"

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

    number_tc = 150
    population = io.loadmat("pop.mat")['population']
    scores = score_population(population, time_metric, number_tc, project)
    population_ids = np.arange(population.shape[0]).astype(int)
    # pareto_front = identify_pareto(scores, population_ids)
    pareto_front = is_pareto_efficient(list(scores))
    # population = population[pareto_front, :]
    # print(population)
    print(pareto_front)
    

if __name__ == "__main__":
    main()
