import os
import pandas as pd
import sys

def read_result(project):
    filePath = os.path.abspath("result/" + project + "_nsga2Select.csv")

    return pd.read_csv(filePath).to_numpy()

def write_data(data):
    tet_filePath = os.path.abspath("evaluation/nsga2_tet_result.txt")
    ms_filePath = os.path.abspath("evaluation/nsga2_ms_result.txt")
    hv_filePath = os.path.abspath("evaluation/nsga2_hv_result.txt")

    # write tet
    with open(tet_filePath, "w") as f:
        for row in data:
            if row[1] == "tet":
                name = row[0]
                f.write(name + "\n")

                temp_data = row[2: 22]

                for i in range(len(temp_data)):
                    if i == len(temp_data) - 1:
                        f.write(str(temp_data[i]) + "\n")
                    else:
                        f.write(str(temp_data[i]) + " ")

                f.write("\n")

    # write ms
    with open(ms_filePath, "w") as f:
        for row in data:
            if row[1] == "ms":
                name = row[0]
                f.write(name + "\n")

                temp_data = row[2: 22]

                for i in range(len(temp_data)):
                    if i == len(temp_data) - 1:
                        f.write(str(temp_data[i]) + "\n")
                    else:
                        f.write(str(temp_data[i]) + " ")

                f.write("\n")

    # write hv
    # with open(hv_filePath, "w") as f:
    #     for row in data:
    #         if row[1] == "hv":
    #             name = row[0]
    #             f.write(name + "\n")
    #
    #             temp_data = row[2: 22]
    #
    #             for i in range(len(temp_data)):
    #                 if i == len(temp_data) - 1:
    #                     f.write(str(temp_data[i]) + "\n")
    #                 else:
    #                     f.write(str(temp_data[i]) + " ")
    #
    #             f.write("\n")

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