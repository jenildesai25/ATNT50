from ATNT50 import ReadData
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier


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
        euclidean_distance = []
        try:
            for i in test_data.values:
                d[test_data.index[count]] = []
                for j in train_data.values:
                    temp = distance.euclidean(j, i)
                    euclidean_distance.append((temp, test_data.index[count]))
                    d[test_data.index[count]].append(temp)
                count += 1
            for k, v in d.items():
                train_data[k] = v
                d[k] = sorted(d[k])
            sorted_euclidean_distance = sorted(euclidean_distance, key=lambda x: x[0])
            return [distance_data[1] for distance_data in sorted_euclidean_distance[:self.k]]
            # return sorted_euclidean_distance
            # return d
        except Exception as e:
            print(e)

    def sort_data_frame(self, sorted_euclidean_distance):
        try:
            # data_frame_1 = data.iloc[:, -2:-1]
            # data_frame_1 = data_frame_1.sort_values(by=['2'], ascending=True)
            # data_frame_2 = data.iloc[:, -3:-2]
            # data_frame_2 = data_frame_2.sort_values(by=['5'], ascending=True)
            # data_frame_3 = data.iloc[:, -4:-3]
            # data_frame_3 = data_frame_3.sort_values(by=['4'], ascending=True)
            # data_frame_4 = data.iloc[:, -5:-4]
            # data_frame_4 = data_frame_4.sort_values(by=['1'], ascending=True)
            # data_frame_5 = data.iloc[:, -1]
            # data_frame_5 = data_frame_5.sort_values(by=['3'], ascending=True)
            k_nearest_neighbour = sorted_euclidean_distance[:self.k]
            k_nearest_neighbour_label = []
            for i in k_nearest_neighbour:
                k_nearest_neighbour_label.append(i[1])
            return k_nearest_neighbour_label
        except Exception as e:
            print(e)

    def find_accuracy(self, test_data, nearest_label):
        # most_occurrence = max(nearest_label, key=nearest_label.count)
        label_value_of_test_data = list(test_data.index)
        correct_label_choice = 0
        for i in label_value_of_test_data:
            if i in nearest_label:
                correct_label_choice += 1
        return (correct_label_choice / float(len(label_value_of_test_data))) * 100.0


if __name__ == '__main__':
    knn_object = Knn(k=3)
    train_data_sample, test_data_sample = knn_object.load_train_test_data("C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/trainDataXY.txt", "C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/testDataXY.txt")
    data_with_euclidean_distance = knn_object.calculate_distance(train_data_sample, test_data_sample)
    nearest_neighbour = knn_object.sort_data_frame(data_with_euclidean_distance)
    print('Accuracy is {}'.format(knn_object.find_accuracy(test_data_sample, nearest_neighbour)))
    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(train_data_sample, train_data_sample.columns)
    # print(neigh.score(test_data_sample, test_data_sample.columns))
