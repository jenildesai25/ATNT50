from ATNT50 import ReadData
from scipy.spatial import distance
import pandas as pd


class Knn:

    def __init__(self, k):
        self.k = k

    def load_train_test_data(self, train_data, test_data):
        try:
            train_data = ReadData.load_data(train_data)
            test_data = ReadData.load_test_data(test_data)
            return train_data, test_data
        except Exception as e:
            print(e)

    def calculate_distance(self, train_data, test_data):
        d = {}
        count = 0
        try:
            for i in test_data.values:
                d[test_data.index[count]] = []
                for j in train_data.values:
                    temp = distance.euclidean(j, i)
                    d[test_data.index[count]].append(temp)
                count += 1
            for k, v in d.items():
                train_data[k] = v
            return train_data
        except Exception as e:
            print(e)

    def sort_data_frame(self, data):
        try:
            data = data.iloc[:, -5:]
            new_data = data.sort_values(['1', '4', '5', '2', '3'], ascending=[True, True, True, True, True])
            print(new_data)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    knn_object = Knn(k=3)
    train_data_sample, test_data_sample = knn_object.load_train_test_data("C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/trainDataXY.txt", "C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/testDataXY.txt")
    data_with_euclidean_distance = knn_object.calculate_distance(train_data_sample, test_data_sample)
    knn_object.sort_data_frame(data_with_euclidean_distance)
