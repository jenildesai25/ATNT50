from Knn import Knn
import Task_E
import LinearRegression
import Svm
import Centroid
from copy import deepcopy

if __name__ == '__main__':

    try:
        # KNN
        train_data_file_name = "../HandWrittenLetters.txt"
        classes_label = 'JLDI'
        numbers = '1245'
        letter_to_digit = Task_E.letter_2_digit_convert(classes_label)
        for i in numbers:
            letter_to_digit.append(i)
        data_frame = Task_E.pickDataClass(train_data_file_name, letter_to_digit)
        train_data_set_without_labels, train_y, test_data_set_without_labels, test_y, train_data_with_labels, test_data_with_labels = Task_E.splitData2TestTrain(data_frame, 39, 9)
        centroid_data_frame_train = deepcopy(train_data_with_labels)
        centroid_data_frame_test = deepcopy(test_data_with_labels)
        k = 5
        knn_object = Knn(k)
        data_with_euclidean_distance = knn_object.calculate_distance(train_data_with_labels.values, test_data_with_labels.values)
        accuracy = knn_object.get_accuracy([(k['Test Label'], k['Classification']) for k in data_with_euclidean_distance])
        print('Accuracy of Knn is:', accuracy)
        # Linear Regression
        linear_regression_object = LinearRegression.LinearRegression()
        N_train, L_train, Xtrain = len(train_y), train_y, train_data_set_without_labels.T

        N_test, Ytest, Xtest = len(test_y), test_y, test_data_set_without_labels.T

        Ytrain = linear_regression_object.indicator_matrix(L_train)
        linear_regression_object.accuracy(N_train, N_test, Xtrain, Xtest, Ytrain, Ytest)

        # SVM
        svm_object = Svm.SupportVectorMachine()
        svm_object.find_accuracy(train_data_set_without_labels, train_y, test_data_set_without_labels, test_y)

        #
        print("Centroid")
        classifier_object = Centroid.Centroid()
        classifier_object.train_data_and_find_mean(centroid_data_frame_train)
        classified_data = classifier_object.classify(centroid_data_frame_test.values)
        cm_accuracy = classifier_object.score(classified_data)
        print('Centroid Method Accuracy:', cm_accuracy)

    except Exception as e:
        print(e.with_traceback())
