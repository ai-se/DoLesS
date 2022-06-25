import numpy as np
import pygmo
import statistics

def is_pareto_efficient(costs, return_mask = True):
    # ref: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    costs = np.array(costs)
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for

    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

# calculate hypervolume
def calcHV(G):
    # GG = []

    # for key, item in enumerate(P):
    #     if item:
    #         GG.append(G[key])
    
    hv = pygmo.hypervolume(G).compute([1] * len(G[0]))

    return hv

# calculate weighted sum of mutation score and normalized test execution time
def calcMSTET(G):
    meanTET = statistics.mean([G[i][0] for i in range(len(G))])
    meanMS = statistics.mean([1-G[i][1] for i in range(len(G))])

    weight_MS_TET = 0.5 * (1 - meanTET) + 0.5 * meanMS

    return weight_MS_TET