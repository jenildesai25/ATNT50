import Task_E
import Centroid
from matplotlib import pyplot

if __name__ == '__main__':
    prediction = []
    classifier_object = Centroid.Centroid()
    # input = input('Alphabets: ')
    input = "hijklmnopq"
    class_ids = Task_E.letter_2_digit_convert(input)
    # print(class_ids)
    splits = [(5, 34), (10, 29), (15, 24), (20, 19), (25, 24), (30, 9), (35, 4)]
    for train, test in splits:
        data = Task_E.pickDataClass("../HandWrittenLetters.txt", class_ids)
        # print(data)
        trainX, trainY, testX, testY, train_data_with_labels, test_data_with_labels = Task_E.splitData2TestTrain(data, 39, test)
        # print(trainX)
        # print(trainY)
        # trainX = Task_E.to_data_frame(trainX, trainY)
        # testX = Task_E.to_data_frame(testX, testY)
        # print(trainX)
        classifier_object.train_data_and_find_mean(train_data_with_labels)
        classified_data = classifier_object.classify(test_data_with_labels.values)
        prediction.append(classifier_object.score(classified_data))
        print(train, test)
    pyplot.plot(prediction, 'b--')
    pyplot.show()
