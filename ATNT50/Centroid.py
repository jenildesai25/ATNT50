from scipy.spatial import distance
import pandas as pd

from ATNT50 import ReadData


class Centroid:

    def __init__(self):
        self.centroids = []

    def train_data_and_find_mean(self, data):
        try:
            data_frame = ReadData.load_data_without_header(data)
            data_frame_values = data_frame.values
            mean_of_train_data = {}
            for train_data_instance in data_frame_values:
                label = train_data_instance[0]
                aggregated_data_record = mean_of_train_data.get(label, {'list_sums': [], 'n': 0})
                if aggregated_data_record['n'] == 0:
                    aggregated_data_record['list_sums'] = train_data_instance[1:]
                    aggregated_data_record['n'] += 1
                else:
                    recorded_sum = aggregated_data_record['list_sums']
                    aggregated_data_record['list_sums'] = [sum(x) for x in zip(recorded_sum, train_data_instance)]
                    aggregated_data_record['n'] += 1
                mean_of_train_data[label] = aggregated_data_record

                for label, values in mean_of_train_data.items():
                    self.centroids.append([x / values['n'] for x in values['list_sums']])
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


if __name__ == '__main__':
    classifier_object = Centroid()
    train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
    test_data_file_name = input('Please insert full file path including drive and directory name for test data: ')
    test_data_set = ReadData.load_data_without_header(test_data_file_name).values
    classifier_object.train_data_and_find_mean(data=train_data_file_name)
    for each_test in test_data_set:
        label = classifier_object.get_calcutated_label(test_instance=each_test)
        print('Predicted Test Label:', each_test[0], 'Calculated Label:', label)