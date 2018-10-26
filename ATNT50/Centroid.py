from ATNT50 import ReadData


class Classifier:

    def find_classifier(self, data, k):
        data_frame = ReadData.load_data(data)


if __name__ == '__main__':
    classifier_object = Classifier()
    classifier_object.find_classifier(data='trainDataXY.txt', k=3)
