"""
This module provides functions for loading and cleaning the data.
"""

import json


def load(file_path):
    """
    Loads and cleans a JSON file of product occurences that are grouped by model ID (as in the TVs.JSON example).

    :param file_path: the file path to a JSON file
    :return: a cleaned list of the data (as if we do not know the model type), and a binary matrix with element (i, j)
    equal to one if item i and item j are duplicates.
    """

    # Load data into dictionary.
    with open(file_path, "r") as file:
        data = json.load(file)

    # Declare common value representations to be replaced by the last value of the list.
    inch = ["Inch", "inches", "\"", "-inch", " inch", "inch"]
    hz = ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", "hz"]
    to_replace = [inch, hz]
    replacements = dict()
    for replace_list in to_replace:
        replacement = replace_list[-1]
        values = replace_list[0:-1]
        for value in values:
            replacements[value] = replacement

    # Clean data.
    clean_list = []
    for model in data:
        for occurence in data[model]:
            # Clean title.
            for value in replacements:
                occurence["title"] = occurence["title"].replace(value, replacements[value])

            # Clean features map.
            features = occurence["featuresMap"]
            for key in features:
                for value in replacements:
                    features[key] = features[key].replace(value, replacements[value])
            clean_list.append(occurence)

    # Compute binary matrix of duplicates, where element (i, j) is one if i and j are duplicates, for i != j, and zero
    # otherwise. Note that this matrix will be symmetric.
    duplicates = [[0] * len(clean_list)] * len(clean_list)
    for i in range(len(clean_list)):
        model_i = clean_list[i]["modelID"]
        for j in range(i + 1, len(clean_list)):
            model_j = clean_list[j]["modelID"]
            if model_i == model_j:
                duplicates[i][j] = 1
                duplicates[j][i] = 1
    return clean_list, duplicates
