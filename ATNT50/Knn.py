import ReadData
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import Task_E


class Knn:

    def __init__(self, k):
        self.k = k

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
        # container is the dictionary that stores all the test label as key and values as 0th index having distances and 1st index as having label of train data.
        container = {}
        # iterate over index so we can store index of test data and store it in container as key.
        count = 0
        # test_data = test_data.T.iloc[:45]
        # train_data = train_data.T.iloc[:150]
        try:
            # i is the test data 1st value. As i increases we are getting immediate next rows as i.
            for i in test_data.values:
                # container[test_data.index[count]] makes the key as container = {'1':[],'2':[],...}. Here '1' is the index of test data.
                container[test_data.values[count][0]] = []
                # index is the variable that holds value for finding label from train data.i.e train_data.index[index] will give me the label name of particular row.
                index = 0
                # j is the train data 1st value. As j increases we are getting immediate next rows as j.
                for j in train_data.values:
                    # temporary variable to store distance between rows between train data and test data.
                    temp = distance.euclidean(j[1:], i[1:])
                    # container[test_data.index[count]].append((temp, train_data.index[index])) append distance, and train data index of that distance
                    container[test_data.values[count][0]].append((temp, j[0]))
                    # index value increases as per row changes for train data.
                    index += 1
                # count increase as train data row finishes and we need another row from test data.
                count += 1
            # this three lines sorting the dictionary and giving sorted distance as first.
            for k, v in container.items():
                train_data[k] = v
                container[k] = sorted(container[k])
            return container
        except Exception as e:
            print(e)

    def sort_data_frame(self, sorted_euclidean_distance):
        """

        :param sorted_euclidean_distance: It is a dictionary that has key as test data label(index is the one that is giving me label as per my case).
        values are sorted ascending. i.e {'1':[(111,1),(123,2),...]} here '1' is the label from test data and values are sorted distance and train data label as (111,1).
        :return: returns k label as key is test data labels and values are k nearest label. i.e {'1':[1,1,1],'2':[2,2,2],...}. here '1' is the test data index(label)
        and [1,1,1] are nearest 3 label values from test data.
        """
        try:
            # dict_of_labels is the dictionary that stores k nearest label as values and test data label as key.
            dict_of_labels = {}
            # iterate over dict_of_labels and store values in it.
            for k, v in sorted_euclidean_distance.items():
                # dict_of_labels[k] is the first index of test data label.
                dict_of_labels[k] = []
                # iterate k times over data and store the label as value.
                for i in range(self.k):
                    # dict_of_labels[k].append(v[:self.k][i][1]) gives you the label from train data.
                    dict_of_labels[k].append(v[:self.k][i][1])
            return dict_of_labels
        except Exception as e:
            print(e)

    def count_matching_label(self, nearest_label):
        """

        :param nearest_label: takes dictionary with having information of nearest label as value and train data index as key.i.e {'1':[1,1,1],'2':[2,2,2],...}
        :return: return highest occurrence of test data index.
        """
        # store value of train data label.
        count = []
        for k, v in nearest_label.items():
            # count.append(max(v, key=v.count)) append highest occurrence of value of key.
            count.append(max(v, key=v.count))
        return count

    def find_accuracy(self, nearest_neighbour_label, occurrence_of_test_data):
        """

        :param nearest_neighbour_label: takes dictionary with having information of nearest label as value and train data index as key.i.e {'1':[1,1,1],'2':[2,2,2],...}
        :param occurrence_of_test_data: takes list of predicted label values for test data.
        :return: print all the required information.
        """
        print('k is: {}\n'.format(self.k))
        print('KNN: \n')
        count = 0
        try:
            # iterate over key and print all the information.
            for k, v in nearest_neighbour_label.items():
                print('Test Label:{} \nNeighbors Label:{} \nClassification: {}\n'.format(k, v, occurrence_of_test_data[count]))
                count += 1
            accuracy_percentage = 0
            count = 0
            # iterate over key and count prediction.
            for k, v in nearest_neighbour_label.items():
                if int(k) == occurrence_of_test_data[count]:
                    accuracy_percentage = accuracy_percentage + (100 // len(nearest_neighbour_label))
                    count += 1
                else:
                    count += 1
            print('Accuracy of Knn is: {}%'.format(accuracy_percentage))
            return accuracy_percentage
        except Exception as e:
            print(e)


if __name__ == '__main__':
    try:
        k = int(input('Please enter the values of k: '))
        train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
        if 'HandWrittenLetters' in train_data_file_name:
            knn_object = Knn(k)
            classes_label = 'ABCDE'
            letter_to_digit = Task_E.letter_2_digit_convert(classes_label)
            data_frame = Task_E.pickDataClass(train_data_file_name, letter_to_digit)
            # train_X, train_Y, test_X, test_Y
            train_data_set, train_y, test_data_set, test_y = Task_E.splitData2TestTrain(data_frame, 39, 9)
            final_train_data = Task_E.append_data_frame_to_label(train_data_set, train_y)
            # final_train_data = final_train_data.transpose()
            final_test_data = Task_E.append_data_frame_to_label(test_data_set, test_y)
            # final_test_data = final_test_data.transpose()
            data_with_euclidean_distance = knn_object.calculate_distance(final_train_data, final_test_data)
            nearest_neighbour = knn_object.sort_data_frame(data_with_euclidean_distance)
            occurrence_label = knn_object.count_matching_label(nearest_neighbour)
            knn_object.find_accuracy(nearest_neighbour, occurrence_label)
        else:
            test_data_file_name = input('Please insert full file path including drive and directory name for test data: ')
            # k_fold = KFold(n_splits=k)
            knn_object = Knn(k)
            # train_data_sample, test_data_sample = knn_object.load_train_test_data(train_data_file_name, test_data_file_name)
            # for train, test in k_fold.split(train_data_sample):
            #     print("Train data:{}\n Test data:{}".format(train, test))
            #     X_train, X_test = train_data_sample[train], train_data_sample[test]
            #     y_train, y_test = train_data_sample[train], train_data_sample[test]
            # print('k split is:', k_fold.get_n_splits(train_data_sample))
            data_with_euclidean_distance = knn_object.calculate_distance(train_data_sample, test_data_sample)
            nearest_neighbour = knn_object.sort_data_frame(data_with_euclidean_distance)
            occurrence_label = knn_object.count_matching_label(nearest_neighbour)
            knn_object.find_accuracy(nearest_neighbour, occurrence_label)
            # print('Accuracy is {}'.format(knn_object.find_accuracy(nearest_neighbour)))
            # neigh = KNeighborsClassifier(n_neighbors=3)
            # neigh.fit(train_data_sample, train_data_sample.index)
            # print(neigh.score(test_data_sample, test_data_sample.index))
    except Exception as e:
        print(e)
