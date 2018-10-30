import pandas as pd


def pickDataClass(file_name, class_ids):
    load_file = pd.read_csv(file_name, sep=",", header=None)

