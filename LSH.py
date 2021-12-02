"""
This module provides LSH functionality, including the extraction of model words and MinHash.
"""

import re

def lsh(data):
    convert_binary(data)


def convert_binary(data):

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
            # (?:^|(?<=[ \[\(])) matches either the start of the string, preceding whitespace, or an opening parenthesis or bracket (exactly once).
            # (?:[0-9]+\.[0-9]+[^0-9\., ()]+) matches any (numeric) - . - (numeric) - (non-numeric) combination (i.e., decimals).
            # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
            # (?:$|(?=[ \)\]])) matches either the end of the string, trailing whitespace, or a closing parenthesis or bracket (exactly once).
            mw_decimal = re.match("(?:^|(?<=[ \[\(]))((?:[0-9]+\.[0-9]+[^0-9\., ()]+)[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))", value)
            if not mw_decimal is None:
                item_mw.append(mw_decimal.group(0))

            # Find binary values, if "Yes" add the key as model word, if "No" add "no" + the key.
            if value == "Yes":
                item_mw.append(key)
            elif value == "No":
                item_mw.append("no" + key)

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
