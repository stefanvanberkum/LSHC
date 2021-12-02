"""
This module provides functions for loading and cleaning the data.
"""

import json


def load(file_path):
    # Load data into dictionary.
    with open(file_path, "r") as file:
        data = json.load(file)

    # Declare common representations to be replaced by the last value of the list.
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
                features["title"] = features["title"].replace(value, replacements[value])

            # Clean features map.
            features = occurence["featuresMap"]
            for key in features:
                for value in replacements:
                    features[key] = features[key].replace(value, replacements[value])
    return data
