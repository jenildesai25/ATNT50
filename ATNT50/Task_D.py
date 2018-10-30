import Task_E
import Centroid
from matplotlib import pyplot

if __name__ == '__main__':
    prediction = []
    classifier_object = Centroid.Centroid()
    # input = input('Alphabets: ')
    input = "klmnopqrst"
    class_ids = Task_E.letter_2_digit_convert(input)
    # print(class_ids)
    for i in range(1, 8):
        data = Task_E.pickDataClass("../HandWrittenLetters.txt", class_ids)
        # print(data)
        trainX, trainY, testX, testY = Task_E.splitData2TestTrain(data.transpose(), 39, i * 5)
        # print(trainX)
        # print(trainY)
        trainX = Task_E.to_data_frame(trainX, trainY)
        testX = Task_E.to_data_frame(testX, testY)
        print(trainX)
        classifier_object.train_data_and_find_mean(trainX.transpose())
        classified_data = classifier_object.classify(testX)
        prediction.append(classifier_object.score(classified_data))
    pyplot.plot(prediction, 'b--')
    pyplot.show()
