# N_train - Number of labels in training data
# N_test - Number of labels  in test data
# Xtrain - Training data excluding labels
# Xtest - Testing data excluding labels
# Ytrain - Indicator matrix of training data
# Ytest - Labels of test data

import numpy as np
import pandas as pd
import ReadData


class LinearRegression:
    @classmethod
    def indicator_matrix(cls, L_train):
        Ytrain = pd.DataFrame()
        unique_Ltrain = set(L_train)
        # print(unique_Ltrain)
        unique_Ltrain_length = unique_Ltrain.__len__()

        column_data = dict()
        for i, label in enumerate(unique_Ltrain):
            column_data[label] = [0] * unique_Ltrain_length
            column_data[label][i] = 1

        for i, label in enumerate(L_train):
            Ytrain[i] = column_data[label]

        # print("Ytrain")
        # print(Ytrain)
        return Ytrain

    @classmethod
    def accuracy(cls, N_train, N_test, Xtrain, Xtest, Ytrain, Ytest):
        # % The following is a Python code for Linear Regression

        A_train = np.ones((1, N_train))  # N_train : number of training instance
        A_test = np.ones((1, N_test))  # N_test  : number of test instance
        Xtrain_padding = np.row_stack((Xtrain, A_train))
        Xtest_padding = np.row_stack((Xtest, A_test))

        # '''computing the regression coefficients'''
        B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain.T)  # (XX')^{-1} X  * Y'  #Ytrain : indicator matrix
        Ytest_padding = np.dot(B_padding.T, Xtest_padding)
        Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1
        err_test_padding = Ytest - Ytest_padding_argmax
        TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / len(err_test_padding)) * 100
        print(TestingAccuracy_padding)


if __name__ == '__main__':
    train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
    test_data_file_name = input('Please insert full file path including drive and directory name for test data: ')
    lg = LinearRegression()
    N_train, L_train, Xtrain = ReadData.load_data(train_data_file_name, "LG")
    # print("Ntrain: " + str(N_train))
    # print("Ltrain: ")
    # print(L_train)
    # print("Xtrain: ")
    # print(Xtrain)
    N_test, Ytest, Xtest = ReadData.load_data(test_data_file_name, "LG")
    # print("Ntest: " + str(N_test))
    # print("Ytest: ")
    # print(Ytest)
    # print("Xtest: ")
    # print(Xtest)
    Ytrain = lg.indicator_matrix(L_train)
    lg.accuracy(N_train, N_test, Xtrain, Xtest, Ytrain, Ytest)
