import pandas as pd
import sys


def load_data(file_name):
    # read_csv file and return data frame.
    load_file = pd.read_csv(file_name, sep=",", header=None)
    load_file = load_file.transpose()
    return load_file


def load_test_data(file_name):
    load_file = pd.read_csv(file_name, sep=",", header=None)
    load_file = load_file.transpose()
    return load_file


def load_data_without_header(file_name):
    load_file = pd.read_csv(file_name, sep=",", header=None)
    load_file = load_file.transpose()
    return load_file


def data_handler(file_name):
    load_file = pd.read_csv(file_name, sep=",", header=None)
    load_file = load_file.transpose()
    train_data = load_file[:30]
    test_data = load_file[30:39]
    return train_data, test_data
# if __name__ == '__main__':
#     # if file is in the same directory where path is just put file name.
#     # if reading file using terminal uncomment next line and pass file_path
#     # file_path = sys.argv[2]
#     # load_data('trainDataXY.txt')
#     load_data('C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/trainDataXY.txt')
