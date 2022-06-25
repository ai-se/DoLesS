from __future__ import division

from pymoo.model.problem import Problem
from functions.function import costFunction

class CSP(Problem):
    def __init__(self, project, time_metric, number_tc):
        super().__init__(n_var=number_tc,
                         n_obj=5,
                         n_constr=0,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True)

        self.project = project
        self.time_metric = time_metric
        self.number_tc = number_tc

    def _evaluate(self, ind, out, *args, **kwargs):
        temp_score = costFunction(ind, self.time_metric, self.number_tc, self.project, [], [], [], [])
        out["F"] = [temp_score[0], 1-temp_score[1], 1-temp_score[2], 1-temp_score[3], 1-temp_score[4]]
