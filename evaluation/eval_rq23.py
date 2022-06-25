import os
import pandas as pd
import sys

def read_result(project):
    filePath = os.path.abspath("result/" + project + "test.csv")

    return pd.read_csv(filePath).to_numpy()

def write_data(data):
    final_tet_filePath = os.path.abspath("evaluation/rq23_tet_result.txt")
    final_ms_filePath = os.path.abspath("evaluation/rq3_ms_result.txt")
    final_derivative_filePath = os.path.abspath("evaluation/rq2_derivative_result.txt")
    final_infinite_filePath = os.path.abspath("evaluation/rq2_infinite_result.txt")
    final_instability_filePath = os.path.abspath("evaluation/rq2_instability_result.txt")
    final_minmax_filePath = os.path.abspath("evaluation/rq2_minmax_result.txt")

    # write tet
    with open(final_tet_filePath, "w") as f:
        temp_nsga2_data = [data[i][0] for i in range(len(data))]
        temp_less_data = [data[i][8] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")


    # write ms
    with open(final_ms_filePath, "w") as f:
        temp_nsga2_data = [data[i][5] for i in range(len(data))]
        temp_less_data = [data[i][13] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")

    # write derivative
    with open(final_derivative_filePath, "w") as f:
        temp_nsga2_data = [data[i][1] for i in range(len(data))]
        temp_less_data = [data[i][9] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")

    # write infinite
    with open(final_infinite_filePath, "w") as f:
        temp_nsga2_data = [data[i][2] for i in range(len(data))]
        temp_less_data = [data[i][10] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")

    # write instability
    with open(final_instability_filePath, "w") as f:
        temp_nsga2_data = [data[i][3] for i in range(len(data))]
        temp_less_data = [data[i][11] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")

    # write minmax
    with open(final_minmax_filePath, "w") as f:
        temp_nsga2_data = [data[i][4] for i in range(len(data))]
        temp_less_data = [data[i][12] for i in range(len(data))]

        f.write("NSGA-2\n")
        for i in range(len(temp_nsga2_data)):
            if i == len(temp_nsga2_data) - 1:
                f.write(str(temp_nsga2_data[i]) + "\n")
            else:
                f.write(str(temp_nsga2_data[i]) + " ")

        f.write("LESS\n")
        for i in range(len(temp_less_data)):
            if i == len(temp_less_data) - 1:
                f.write(str(temp_less_data[i]) + "\n")
            else:
                f.write(str(temp_less_data[i]) + " ")

def main(project):
    os.chdir("..")

    data = read_result(project)
    write_data(data)


if __name__ == "__main__":
    print("usage:")
    print("-p [project]: clean the results of that project")
    print("output:")
    print("tet_result.txt - The cleaned text file for test execution time comparison.")
    print("ms_result.txt - The cleaned text file for mutation score comparison.")
    print("hv_result.txt - The cleaned text file for hypervolumn indicator comparison.")

    if len(sys.argv) <= 1:
        print("please specify one project")
    else:
        if "-p" in sys.argv:
            project = sys.argv[sys.argv.index("-p")+1]
            main(project)
        else:
            print("please use -p command to enter the project name")