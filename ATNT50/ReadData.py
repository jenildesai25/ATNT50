import pandas as pd
import sys

class LoadData:

    def load_data(self, file_name):
        pass


if __name__ == '__main__':
    read_load_data = LoadData()
    # if file is in the same directory where path is just put file name.
    # if reading file using terminal uncomment next line and pass file_path 
    # file_path = sys.argv[2]
    read_load_data.load_data('trainDataXY.txt')
