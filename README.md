# Test_Selection_2021

This repo contains the reproduction scripts for the study 

**How to Find Small Test Suites for Multi-goal Cyber-Physical Problems (a DoLesS approach)**

## Usage

### RQ1
For RQ1, we replicated the experiments from Arrieta et al.'s study: "Pareto efficient multi-objective black-box test case selection for
simulation-based testing"

To run the replication experiments, please follow the following steps:

- First, execute NSGA-III algorithm for all case studies by
```
$ python3 nsga3_runner.py -p [project name]
```

- Second, execute MOEA/D algorithm for all case studies by
```
$ python3 moead_runner.py -p [project name]
```

- Third, execute NSGA-II algorithm for all case studies by
```
$ python3 nsga3_runner.py -p [project name]
```

- Fourth, find the best subset of objectives for NSGA-II algorithm by following the guidelines in **evaluation** folder.
- Fifth, compare NSGA-II, NSGA-III, MOEA/D by following the guidelines in **evaluation** folder.

### RQ2
For RQ3, we compare the performance of our method with NSGA-II approach in terms of two evaluation metrics.

To obtain the results for RQ3, please following the guidelines in **evaluation** folder. All necessary data is obtained in previous RQs.

### RQ3
For RQ4, we compare the run time of our approach with NSGA-II approach. 

### DISCUSSION
For discussion section, the effectiveness measurement values can be obtained by following steps:

- First, execute NSGA-II algorithm and DoLesS algorithm by
```
$ python3 rqs.py -p [project name] -f [best combination of objectives]
```
Please note that **[best combination of objectives]** should follow the format **[name of f1],[name of f2],[name of f3]** (e.g. time,instability,minmax)

The name of objectives are: (a) time (for test exeuction time) (b) discontinuity (for discontinuity anti-pattern) 
(c) infinite (for growth to infinity anti-pattern) (d) instability (for instability anti-pattern) and (e) minmax (for difference of minimum and maximum difference).

- Second, compare NSGA-II approach and our DoLesS approach by following the guidelines in **evaluation** folder.

The graphs can be reproduced by
```
$ python3 discussion.py -p [project name]
```
