import os
import pandas as pd
import sys

def read_nsga3_result(project):
    filePath = os.path.abspath("result/" + project + "_nsga3.csv")

    return pd.read_csv(filePath).to_numpy()

def read_moead_result(project):
    filePath = os.path.abspath("result/" + project + "_moead.csv")

    return pd.read_csv(filePath).to_numpy()

def read_nsga2_result(project):
    filePath = os.path.abspath("result/" + project + "test.csv")

    return pd.read_csv(filePath).to_numpy()

def write_data(data_nsga3, data_nsga2, data_moead):
    rq1_tet_filePath = os.path.abspath("evaluation/rq1_tet_result.txt")
    rq1_ms_filePath = os.path.abspath("evaluation/rq1_ms_result.txt")

    # write tet
    with open(rq1_tet_filePath, "w") as f:
        temp_nsga2_data = [data_nsga2[i][0] for i in range(len(data_nsga2))]
        temp_nsga3_data = [data_nsga3[i][0] for i in range(len(data_nsga3))]
        temp_moead_data = [data_moead[i][0] for i in range(len(data_moead))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("NSGA-3\n")
        for i in range(len(temp_nsga3_data)):
            if i == len(temp_nsga3_data) - 1:
                f.write(str(temp_nsga3_data[i]) + "\n")
            else:
                f.write(str(temp_nsga3_data[i]) + " ")

        f.write("MOEA/D\n")
        for i in range(len(temp_moead_data)):
            if i == len(temp_moead_data) - 1:
                f.write(str(temp_moead_data[i]) + "\n")
            else:
                f.write(str(temp_moead_data[i]) + " ")

    # write ms
    with open(rq1_ms_filePath, "w") as f:
        temp_nsga2_data = [data_nsga2[i][5] for i in range(len(data_nsga2))]
        temp_nsga3_data = [data_nsga3[i][5] for i in range(len(data_nsga3))]
        temp_moead_data = [data_moead[i][5] for i in range(len(data_moead))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("NSGA-3\n")
        for i in range(len(temp_nsga3_data)):
            if i == len(temp_nsga3_data) - 1:
                f.write(str(temp_nsga3_data[i]) + "\n")
            else:
                f.write(str(temp_nsga3_data[i]) + " ")

        f.write("MOEA/D\n")
        for i in range(len(temp_moead_data)):
            if i == len(temp_moead_data) - 1:
                f.write(str(temp_moead_data[i]) + "\n")
            else:
                f.write(str(temp_moead_data[i]) + " ")

def main(project):
    os.chdir("..")

    data_nsga2 = read_nsga2_result(project)
    data_nsga3 = read_nsga3_result(project)
    data_moead = read_moead_result(project)
    write_data(data_nsga3, data_nsga2, data_moead)


if __name__ == "__main__":
    print("usage:")
    print("-p [project]: clean the results of that project")
    print("output:")
    print("tet_result.txt - The cleaned text file for test execution time comparison.")
    print("ms_result.txt - The cleaned text file for mutation score comparison.")

    if len(sys.argv) <= 1:
        print("please specify one project")
    else:
        if "-p" in sys.argv:
            project = sys.argv[sys.argv.index("-p")+1]
            main(project)
        else:
            print("please use -p command to enter the project name")