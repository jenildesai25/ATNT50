from Knn import Knn
import Task_E
import LinearRegression
import Svm

if __name__ == '__main__':

    try:
        # KNN
        k = int(input('Please enter the values of k: '))

        # train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
        train_data_file_name = "../HandWrittenLetters.txt"
        classes_label = 'ABCDE'
        letter_to_digit = Task_E.letter_2_digit_convert(classes_label)
        data_frame = Task_E.pickDataClass(train_data_file_name, letter_to_digit)
        train_data_set_without_labels, train_y, test_data_set_without_labels, test_y, train_data_with_labels, test_data_with_labels = Task_E.splitData2TestTrain(data_frame, 39, 9)

        knn_object = Knn(k)
        data_with_euclidean_distance = knn_object.calculate_distance(train_data_with_labels, test_data_with_labels)
        nearest_neighbour = knn_object.sort_data_frame(data_with_euclidean_distance)
        occurrence_label = knn_object.count_matching_label(nearest_neighbour)
        knn_object.find_accuracy(nearest_neighbour, occurrence_label)

        # Linear Regression
        linear_regression_object = LinearRegression.LinearRegression()
        N_train, L_train, Xtrain = len(train_y), train_y, train_data_set_without_labels.T
        # print("Ntrain: " + str(N_train))
        # print("Ltrain: ")
        # print(L_train)
        # print("Xtrain: ")
        # print(Xtrain)
        N_test, Ytest, Xtest = len(test_y), test_y, test_data_set_without_labels.T
        # print("Ntest: " + str(N_test))
        # print("Ytest: ")
        # print(Ytest)
        # print("Xtest: ")
        # print(Xtest)
        Ytrain = linear_regression_object.indicator_matrix(L_train)
        linear_regression_object.accuracy(N_train, N_test, Xtrain, Xtest, Ytrain, Ytest)

        # SVM
        svm_object = Svm.SupportVectorMachine(train_data_set_without_labels, train_y, test_data_set_without_labels, test_y)

    except Exception as e:
        print(e)
