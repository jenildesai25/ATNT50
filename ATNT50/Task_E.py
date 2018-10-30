import pandas as pd
import numpy as np


def pickDataClass(filename, class_ids):
    load_file = pd.read_csv(filename, sep=",", header=None)
    load_file = load_file.transpose()
    result = []
    for i in class_ids:
        for j in load_file.values:
            if int(j[0]) == int(i):
                result.append(j)
    result = pd.DataFrame(result)
    return result


def splitData2TestTrain(filename, number_per_class, test_instances):
    try:
        if '.txt' in str(filename):
            load_file = pd.read_csv(filename, sep=",", header=None)
            load_file = load_file.transpose()
            test_X = []  # Test data without labels
            test_Y = []  # Test data labels only
            train_X = []  # Training data without labels
            train_Y = []  # Training data labels only
            filename = filename.transpose()
            test_instance_count = dict()
            for col_number in load_file.columns:
                val = load_file[col_number].values
                label = val[0]
                current_count = test_instance_count.get(label, 0)
                if current_count < test_instances:
                    # Add to test
                    test_Y.append(label)
                    test_X.append(val[1:])
                else:
                    # Add to training
                    train_Y.append(label)
                    train_X.append(val[1:])
                current_count += 1
                test_instance_count[label] = current_count
            train_X = pd.DataFrame(train_X)
            test_X = pd.DataFrame(train_Y)
            return train_X, train_Y, test_X, test_Y
        else:
            # we know that this is dataframe.
            test_X = []  # Test data without labels
            test_Y = []  # Test data labels only
            train_X = []  # Training data without labels
            train_Y = []  # Training data labels only
            filename = filename.transpose()
            test_instance_count = dict()
            for col_number in filename.columns:
                val = filename[col_number].values
                label = val[0]
                current_count = test_instance_count.get(label, 0)
                if current_count < test_instances:
                    # Add to test
                    test_Y.append(label)
                    test_X.append(val[1:])
                else:
                    # Add to training
                    train_Y.append(label)
                    train_X.append(val[1:])
                current_count += 1
                test_instance_count[label] = current_count
            train_X = pd.DataFrame(train_X)
            test_X = pd.DataFrame(test_X)
            return train_X, train_Y, test_X, test_Y
    except Exception as e:
        print(e)


def append_data_frame_to_label(data, label):
    temp_X = data.values
    temp_XY = list()
    temp_XY.append(label)
    for array in temp_X:
        temp_XY.append(list(array))
    return pd.DataFrame(temp_XY)


# X: Dataframe of training or test data
# Y: Dataframe/row of header/labels of training or test data
# filename: Name of file to save without extension
def store(X, Y, filename):
    append_data_frame_to_label(X, Y).to_csv(filename + ".txt", sep=',')


# letter: String of alphabets
def letter_2_digit_convert(letters):
    return [ord(character) - 96 for character in letters.lower()]

# if __name__ == '__main__':
#     file = input('Please insert full file path including drive and directory name for train data: ')
#     index_values = list(input('Please insert label that you need to select for data:'))
#     if file:
#         data = pickDataClass(file, index_values)
#         print(data)
#     else:
#         pass
#     train_data_file_name = input('Please insert full file path including drive and directory name for train data: ')
#     number_per_classes = int(input('Please insert the number of images each instance have: '))
#     test_instance = int(input('Please insert the number of images you need for test data: '))
#     if train_data_file_name:
#         print(splitData2TestTrain(train_data_file_name, number_per_classes, test_instance))
#     else:
#         print(splitData2TestTrain(data, number_per_classes, test_instance))
