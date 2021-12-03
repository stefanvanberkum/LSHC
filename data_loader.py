"""
This module provides functions for loading and cleaning the data.
"""

import json

def load(file_path):
    """
    Loads and cleans a JSON file of product occurences that are grouped by model ID (as in the TVs.JSON example).

    :param file_path: the file path to a JSON file
    :return: a cleaned and shuffled list of the data (as if we do not know the model type), and a copy of the cleaned
    data.
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
    return clean_list, data
