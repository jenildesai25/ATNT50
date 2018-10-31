import ReadData
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import Task_E


class Knn:

    def __init__(self, k, distance_method):
        self.k = k
        self.distance_method = distance_method

    def load_train_test_data(self, train_data, test_data):
        """

        :param train_data: train data(raw data) is file we wanted our algorithm to train with so we can use that result with test data.
        :param test_data: test data(raw data) for checking that our prediction is right or not and finding the accuracy.
        :return: well formed train and test data with having rows as one image and index is label of the image.
        """
        try:
            # next line will give you transposed and well formatted train data.
            train_data = ReadData.load_data(train_data)
            # next line will give you transposed and well formatted test data.
            test_data = ReadData.load_test_data(test_data)
            return train_data, test_data
        except Exception as e:
            print(e)

    def calculate_distance(self, train_data, test_data):
        """

        :param train_data: formatted train data that we get from load_train_test_data function.
        :param test_data: formatted test data that we get from load_train_test_data function.
        :return: dictionary with having key as test data index i.e in our case ['1','4','5','3','2']. be careful keys are in string.
        and values are sorted euclidean distance with label from train data.
        container = {'1':[(1111,1),(131241,3),...]}
        '1' is the key from test data and (1111) is the distance between 1st instance of train data with 1st data instance from test data.
        1 is the label of train data.
        """
        result_list = list()
        for test_data_instance in test_data:
            result_dict = dict()
            # print('test_data_instance',test_data_instance)
            nearest_neighbors = self.get_nearest_neighbors(train_data, test_data_instance)
            # print('nearest_neighbors',nearest_neighbors)
            calculated_classification = self.get_classification(nearest_neighbors)
            result_dict['Test Label'] = test_data_instance[0]
            result_dict['Neighbors Label'] = nearest_neighbors
            result_dict['Classification'] = calculated_classification
            result_list.append(result_dict)  # Given Classification, Calculated Classification

        # Calculate Accuracy
        return result_list

    def get_nearest_neighbors(self, training_data, testing_data):
        try:
            distances = []
            for training_data in training_data:
                if self.distance_method == "eu":
                    euclidean_dist = distance.euclidean(training_data[1:], testing_data[1:])
                elif self.distance_method == "cos":
                    euclidean_dist = distance.cosine(training_data[1:], testing_data[1:])
                distances.append((euclidean_dist, training_data[0]))

            # Sort by distances
            sorted_distances = sorted(distances, key=lambda x: x[0])

            return [distance_data[1] for distance_data in sorted_distances[:self.k]]

        except Exception as e:
            print(e)

    def get_classification(self, nearest_neighbour):
        """ Returns label
                """
        class_votes = dict()
        for label in nearest_neighbour:
            vote = class_votes.get(label, 0)
            class_votes[label] = vote + 1

        sorted_votes = sorted(
            list(class_votes.items()), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]

    def get_accuracy(self, prediction_n_test):
        correct = 0
        for prediction, test in prediction_n_test:
            if prediction == test:
                correct += 1
        return (float(correct) / float(len(prediction_n_test))) * 100.0