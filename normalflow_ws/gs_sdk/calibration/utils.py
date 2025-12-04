import csv
import numpy as np


def load_csv_as_dict(csv_path):
    """
    Load the csv file entries as dictionaries.

    :params csv_path: str; the path of the csv file.
    :returns: dict; the dictionary of the csv file.
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        data = list(reader)
        keys = reader.fieldnames
        data_dict = {}
        for key in keys:
            data_dict[key] = []
        for line in data:
            for key in keys:
                data_dict[key].append(line[key])
    return data_dict
