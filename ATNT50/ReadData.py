import pandas as pd
import sys


def load_data(file_name):
    # read_csv file and return data frame.
    load_file = pd.read_csv(file_name, sep=",")
    column_index = [i for i in range(45)]
    new_column_values = []
    for i in range(1, 6):
        for j in range(1, 10):
            new_column_values.append(i)
    for i in range(0, len(column_index)):
        load_file.columns._data[i] = new_column_values[i]
    load_file = load_file.transpose()
    return load_file


def load_test_data(file_name):
    load_file = pd.read_csv(file_name, sep=",")
    load_file = load_file.transpose()
    return load_file


def load_data_without_header(file_name):
    load_file = pd.read_csv(file_name, sep=",", header=None)
    load_file = load_file.transpose()
    return load_file
# if __name__ == '__main__':
#     # if file is in the same directory where path is just put file name.
#     # if reading file using terminal uncomment next line and pass file_path
#     # file_path = sys.argv[2]
#     # load_data('trainDataXY.txt')
#     load_data('C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/trainDataXY.txt')
