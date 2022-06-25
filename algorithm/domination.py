import math

from functions.function import generate_scores

class Domination(object):
    def __init__(self, ind):
        self.ind = ind

    def __lt__(self, other):
        s1, s2, n = 0, 0, len(self.ind)

        for idx, item in enumerate(self.ind):
            if idx == 0:
                w = -1
            else:
                w = 1

            a = self.ind[idx]
            b = other.ind[idx]
            s1 -= math.e ** (w * (a - b) / n)
            s2 -= math.e ** (w * (b - a) / n)

        return s1 / n < s2 / n

def domination_runner(candidates):
    temp_list = []

    for item in candidates:
        temp_list.append(Domination(item[1]))

    temp_list = sorted(temp_list)

    # convert Domination object to normal list
    count = 0
    result_list = []
    for item in temp_list:
        result_list.append(vars(item)['ind'])
        count += 1

        if count >= 100:
            break

    return result_list


# pop_size = 10000
# scores = generate_scores(pop_size)
# new_res = domination_runner(scores)
#
# count = 0
# final_res = []
# while count < 100:
#     final_res.append(new_res[count])
#     count += 1
#
# print(final_res)



# d1 = Domination([0.3, 0.6, 0.5])
# d2 = Domination([0.3, 0.8, 0.65])
# d3 = Domination([0.45, 0.5, 0.5])
# d4 = Domination([0.35, 0.6, 0.6])
#
# # d1, d2, d3, d4 = [0.3, 0.7, 0.65], [0.3, 0.8, 0.65], [0.45, 0.5, 0.5], [0.35, 0.6, 0.6]
#
# list = [d1, d2, d3, d4]
# list = sorted(list)
#
# for item in list:
#     print(vars(item))
