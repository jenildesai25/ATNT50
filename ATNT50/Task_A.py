from Knn import Knn

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
