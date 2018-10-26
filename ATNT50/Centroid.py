from datetime import date

from ATNT50 import ReadData


class Centroid:

    def find_classifier(self, data):
        data_frame = ReadData.load_data(data)
        data_frame = data_frame.transpose()
        label_name = list(data_frame._info_axis)
        label_name_count = {}
        count = 0
        for i in label_name:
            if i not in label_name_count:
                count = 1
                label_name_count[i] = 1
            else:
                label_name_count[i] = count + 1
                count = count + 1
        mean = {}
        key = 0
        for i in label_name_count.values():
            index = [0, 9]
            mean[list(label_name_count.keys())[key]] = []
            for j in range(data_frame.shape[0]):
                values = data_frame.values[index[0], :index[1]]
                index[0] = index[0] + 1
                mean[list(label_name_count.keys())[key]].append(values.mean())

            key = key + 1


if __name__ == '__main__':
    classifier_object = Centroid()
    classifier_object.find_classifier(data='C:/Users/jenil/OneDrive - University of Texas at Arlington/UTA/sem 3/CSE 5334/Project1/ATNT50/trainDataXY.txt')
