from Task_E import data_to_frame
from Knn import Knn
from Svm import SupportVectorMachine
import LinearRegression
import Centroid
import pandas as pd
from sklearn.model_selection import KFold
from copy import deepcopy

if __name__ == '__main__':
    file_name = "../ATNTFaceImages400.txt"
    master_data_frame = pd.read_csv(file_name, sep=",", header=None)
    main_data = master_data_frame.transpose().values
    labels = main_data[0:, 0]
    data_with_out_label = main_data[0:, 1:]

    kfold_val = 3
    kf = KFold(n_splits=kfold_val, random_state=None, shuffle=True)

    kf.get_n_splits(data_with_out_label)
    knn_accuracies = []
    cm_accuracies = []
    lrm_accuracies = []
    svm_accuracies = []
    for train_index, test_index in kf.split(data_with_out_label):
        trainX, testX = data_with_out_label[train_index], data_with_out_label[test_index]
        trainY, testY = labels[train_index], labels[test_index]
        training_data_frame = data_to_frame(pd.DataFrame(trainX).T, trainY)
        test_data_frame = data_to_frame(pd.DataFrame(testX).T, testY)
        training_data_frame_in_row_with_labels = training_data_frame.T
        test_data_frame_in_row_with_labels = test_data_frame.T
        centroid_data_frame_train = deepcopy(training_data_frame_in_row_with_labels)
        centroid_data_frame_test = deepcopy(test_data_frame_in_row_with_labels)
        # KNN
        k = 3
        knn_object = Knn(k)
        data_with_euclidean_distance = knn_object.calculate_distance(training_data_frame_in_row_with_labels
                                                                     , test_data_frame_in_row_with_labels)
        nearest_neighbour = knn_object.sort_data_frame(data_with_euclidean_distance)
        occurrence_label = knn_object.count_matching_label(nearest_neighbour)
        accuracy = knn_object.find_accuracy(nearest_neighbour, occurrence_label)
        knn_accuracies.append(accuracy)

        # Linear Regression
        linear_regression_object = LinearRegression.LinearRegression()
        train_y = training_data_frame.iloc[0]
        train_data_set_without_labels = training_data_frame.iloc[1:]
        N_train, L_train, Xtrain = len(train_y), train_y, train_data_set_without_labels
        test_y = test_data_frame.iloc[0]
        test_data_set_without_labels = test_data_frame.iloc[1:]
        N_test, Ytest, Xtest = len(test_y), test_y, test_data_set_without_labels
        Ytrain = linear_regression_object.indicator_matrix(L_train)
        accuracy_val = linear_regression_object.accuracy(N_train, N_test, Xtrain, Xtest, Ytrain, Ytest)
        lrm_accuracies.append(accuracy_val)

        # SVM
        svm_object = SupportVectorMachine()
        score = svm_object.find_accuracy(train_data_set_without_labels.T, list(train_y.values), test_data_set_without_labels.T, list(test_y.values))
        svm_accuracies.append(score)

        # Centroid
        centroid_object = Centroid.Centroid()
        print("Centroid ")
        centroid_object.train_data_and_find_mean(centroid_data_frame_train)
        classified_data = centroid_object.classify(centroid_data_frame_test.values)
        cm_accuracy = centroid_object.score(classified_data)
        print('Centroid Method Accuracy:', cm_accuracy)
        cm_accuracies.append(cm_accuracy)
        # for each_test in centroid_data_frame_test.values:
        #     label = centroid_object.get_calcutated_label(test_instance=each_test)
        #     print('Predicted Test Label:', each_test[0], 'Calculated Label:', label)
    print('KNN Avg accuracy:', sum(knn_accuracies) / len(knn_accuracies))
    print('Centroid Avg accuracy:', sum(cm_accuracies) / len(cm_accuracies))
    print('Linear Regression Avg accuracy:', sum(lrm_accuracies) / len(lrm_accuracies))
    print('SVM Avg accuracy:', sum(svm_accuracies) / len(lrm_accuracies))