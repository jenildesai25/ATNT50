from sklearn.svm import SVC
import ReadData

if __name__ == '__main__':
    print("Please insert full file path including drive and directory name for: ")
    train_data_file_name = input('Training data: ')
    test_data_file_name = input('Test data: ')

    train_labels, train_data = ReadData.load_data(train_data_file_name, "SVM")
    test_labels, test_data = ReadData.load_data(test_data_file_name, "SVM")

    SVM = SVC()
    SVM.fit(train_data, train_labels)
    SVM.predict(test_data)
    score = SVM.score(test_data, test_labels)
    print(score)
