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
        # (?:$|(?=[ \)\]])) matches either the end of the string, trailing whitespace (exactly once), or a closing parenthesis or bracket (exactly once).
        mw_title = re.findall("(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:[0-9]+\.[0-9]+[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))", item["title"])

        for mw in mw_title:
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

        # Find model words in the key-value pairs.

    return binary_vec

def minhash(data):
    print()
