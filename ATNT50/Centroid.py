from scipy.spatial import distance
import pandas as pd
import ReadData


class Centroid:

    def __init__(self):
        self.centroids = []

    def train_data_and_find_mean(self, data):
        """

        :param data: takes train data in row format. that means every row is an image and 1st element of row is label and after that it
        will be data. For Example: data frame should be in [1,2,3,4,5,6,7,8,9] here 1 is the label and after that [2,3,4,5,6,7,8,9] is the data.
        :return: append values to self.centroid.
        """
        try:
            data_frame = data
            data_frame_values = data_frame.values
            mean_of_train_data = {}
            for train_data_instance in data_frame_values:
                label = train_data_instance[0]
                aggregated_data_record = mean_of_train_data.get(label, {'list_sums': [], 'n': 0})
                if aggregated_data_record['n'] == 0:
                    aggregated_data_record['list_sums'] = train_data_instance
                    aggregated_data_record['n'] += 1
                else:
                    recorded_sum = aggregated_data_record['list_sums']
                    aggregated_data_record['list_sums'] = [sum(x) for x in zip(recorded_sum, train_data_instance)]
                    aggregated_data_record['n'] += 1
                mean_of_train_data[label] = aggregated_data_record

            for label, values in mean_of_train_data.items():
                self.centroids.append([x / values['n'] for x in values['list_sums']])
            # print(self.centroids)
        except Exception as e:
            print(e)

    def get_calcutated_label(self, test_instance):
        """Returns the label of the closest centroid
        """

        euclidean_distances = []

        for each_centroid in self.centroids:
            euclidean_dist = distance.euclidean(each_centroid[1:], test_instance[1:])
            euclidean_distances.append((each_centroid[0], euclidean_dist))

        sorted_distances = sorted(euclidean_distances, key=lambda x: x[1])

        return sorted_distances[0][0]

    def classify(self, test_data_set):
        """

        :param test_data_set: takes test data as input. data is row formatted. 1st row contains 1st image and 1st element is class label
        and after that it contains data.For Example: data frame should be in [1,2,3,4,5,6,7,8,9] here 1 is the label and after
        that [2,3,4,5,6,7,8,9] is the data.
        :return:
        """
        prediction = []
        for test_data in test_data_set:
            label = self.get_calcutated_label(test_data)
            prediction.append((test_data[0], label))  # (Label given in file, Label calculated from algorithm)
        return prediction

    def score(self, classified_data):
        """

        :param classified_data: take prediction count score
        :return: final score
        """
        label_match = 0
        for label_file, label_calculated in classified_data:
            if label_file == label_calculated:
                label_match += 1
        return (float(label_match) / float(len(classified_data))) * 100.0


if __name__ == '__main__':
    classifier_object = Centroid()
    train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
    if 'HandWrittenLetters' in train_data_file_name:
        train_data_set, test_data_set = ReadData.data_handler(train_data_file_name)
        # test_data_set = ReadData.load_data_without_header(test_data_file_name).values
        classifier_object.train_data_and_find_mean(data=train_data_set)
        # classifier_object.train_data_and_find_mean(data=train_data_file_name)

        classified_data = classifier_object.classify(test_data_set)
        # for each_test in test_data_set.values:
        #     label = classifier_object.get_calcutated_label(test_instance=each_test)
        #     print('Predicted Test Label:', each_test[0], 'Calculated Label:', label)
        prediction = classifier_object.score(classified_data)
        print("Prediction: " + str(prediction))

    else:
        test_data_file_name = input('Please insert full file path including drive and directory name for test data: ')
        test_data_set = ReadData.load_data(test_data_file_name).values
        train_data_file = ReadData.load_data(train_data_file_name)
        classifier_object.train_data_and_find_mean(data=train_data_file)
        for each_test in test_data_set:
            label = classifier_object.get_calcutated_label(test_instance=each_test)
            print('Predicted Test Label:', each_test[0], 'Calculated Label:', label)
