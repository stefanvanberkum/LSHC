"""
This module provides LSH functionality, including the extraction of model words and MinHash.
"""

import re

def lsh(data):
    convert_binary(data)


def lsh_old(data):
    convert_binary_old(data)


def convert_binary(data):
    """
    Transforms a list of items to a binary vector product representation, using model words in the title, decimals in
    the feature values, and binary/numeric feature keys.

    :param data: a list of items
    :return: a binary vector product representation
    """

    # List of common features.
    common_features = ["Component", "HDMI", "USB", "Composite", "Energy Star", "Smart", "PC Input"]

    # For computational efficiency, we keep all model words as keys in a dictionary, where its value is the
    # corresponding row in the binary vector product representation.
    model_words = dict()
    binary_vec = []

    # Loop through all items to find model words.
    for i in range(len(data)):
        item = data[i]
        # Find model words in the title.
        # (?:^|(?<=[ \[\(])) matches either the start of the string, preceding whitespace, or an opening parenthesis or bracket (exactly once).
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        # (?:[0-9]+[^0-9\., ()]+) matches any (numeric) - (non numeric) combination.
        # (?:[^0-9\., ()]+[0-9]+) matches any (non numeric) - (numeric) combination.
        # (?:[0-9]+\.[0-9]+[^0-9\., ()]+) matches any (numeric) - . - (numeric) - (non-numeric) combination (i.e., decimals).
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        # (?:$|(?=[ \)\]])) matches either the end of the string, trailing whitespace, or a closing parenthesis or bracket (exactly once).
        mw_title = re.findall("(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:[0-9]+\.[0-9]+[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))", item["title"])
        item_mw = mw_title

        # Find model words in the key-value pairs.
        features = item["featuresMap"]
        for key in features:
            value = features[key]

            # Find decimals.
            # ([0-9]+\.[0-9]+) matches any (numeric) - . - (numeric) - (non-numeric) combination (i.e., decimals).
            # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
            mw_decimal = re.findall("([0-9]+\.[0-9]+)[a-zA-Z]*", value)
            for decimal in mw_decimal:
                item_mw.append(decimal)

            # Group some common binary and numeric features.
            key_mw = key
            for feature in common_features:
                if feature.lower() in key.lower():
                    key_mw = feature
                    break

            # Find binary values, if "Yes" add the key as model word, if "No" add "no" + the key.
            one_or_more = re.fullmatch("[1-9][0-9]*", value)
            if value == "Yes" or one_or_more is not None:
                item_mw.append(key_mw)
            elif value == "No" or value == 0:
                item_mw.append("no" + key_mw)

        # Loop through all identified model words and update the binary vector product representation.
        for mw in item_mw:
            if mw in model_words:
                # Set index for model word to one.
                row = model_words[mw]
                binary_vec[row][i] = 1
            else:
                # Add model word to the binary vector, and set index to one.
                binary_vec.append([0] * len(data))
                binary_vec[len(binary_vec) - 1][i] = 1

                # Add model word to the dictionary.
                model_words[mw] = len(binary_vec) - 1

    return binary_vec


def convert_binary_old(data):
    """
    Transforms a list of items to a binary vector product representation, using model words in the title and decimals in
    the feature values.

    NOTE. This is the old implementation by Hartveld et al. (2018), implemented for evaluation purposes.

    :param data: a list of items
    :return: a binary vector product representation
    """

    # For computational efficiency, we keep all model words as keys in a dictionary, where its value is the
    # corresponding row in the binary vector product representation.
    model_words = dict()
    binary_vec = []

    # Loop through all items to find model words.
    for i in range(len(data)):
        item = data[i]
        # Find model words in the title.
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        # (?:[0-9]+[^0-9, ]+) (incorrectly) matches any (numeric) - (non numeric) combination.
        # (?:[^0-9, ]+[0-9]+) (incorrectly) matches any (non numeric) - (numeric) combination.
        # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
        mw_title = re.findall("([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", item["title"])
        item_mw = mw_title

        # Find model words in the key-value pairs.
        features = item["featuresMap"]
        for key in features:
            value = features[key]

            # Find decimals.
            # (?:(^[0-9]+(?:\.[0-9]+))[a-zA-Z]+$) matches any (numeric) - . - (numeric) - (non-numeric) combination (i.e., decimals).
            # (^[0-9](?:\.[0-9]+)$)) matches any (numeric) - . - (numeric) combination (i.e., decimals).
            # [a-zA-Z0-9]+ matches any alphanumeric character (one or more times).
            mw_decimal = re.findall("(?:(?:(^[0-9]+(?:\.[0-9]+))[a-zA-Z]+$)|(^[0-9](?:\.[0-9]+)$))", value)
            for decimal in mw_decimal:
                for group in decimal:
                    if group != "":
                        item_mw.append(group)

        # Loop through all identified model words and update the binary vector product representation.
        for mw in item_mw:
            if mw in model_words:
                # Set index for model word to one.
                row = model_words[mw]
                binary_vec[row][i] = 1
            else:
                # Add model word to the binary vector, and set index to one.
                binary_vec.append([0] * len(data))
                binary_vec[len(binary_vec) - 1][i] = 1

                # Add model word to the dictionary.
                model_words[mw] = len(binary_vec) - 1

    return binary_vec


def minhash(data):
    print()


def common_binary(data):
    feature_count = dict()

    # Loop through all items to identify common binary features.
    for i in range(len(data)):
        item = data[i]
        features = item["featuresMap"]

        for key in features:
            value = features[key]

            numeric = re.match("^[0-9]+$", value)
            if value == "Yes" or value == "No" or numeric is not None:
                if key in feature_count:
                    feature_count[key] += 1
                else:
                    feature_count[key] = 1

    count_list = [(v,k) for k,v in feature_count.items()]
    count_list.sort(reverse=True)
    for feature in count_list:
        print(feature[1], feature[0])
