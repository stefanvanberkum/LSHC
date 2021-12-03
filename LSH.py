"""
This module provides LSH functionality, including the extraction of model words and MinHash.
"""

import re
from sympy import nextprime
import random
import numpy as np

def lsh(data):
    binary_vec = convert_binary(data)
    signature = minhash(binary_vec, round(0.5 * len(binary_vec)))


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


def minhash(binary_vec, n):
    """
    Computes a MinHash signature matrix using n random hash functions, which will result in an n x c signature matrix,
    where c is the number of columns (items). These random hash functions are of the form (a + bx) mod k, where a and b
    are randomly generated integers, k is the smallest prime number that is larger than or equal to r (the original
    number of rows in the r x c binary vector). We use quick, vectorized, numpy operations to substantially reduce
    computation time.

    :param binary_vec: a binary vector product representation
    :param n: the number of rows in the new signature matrix
    :return: the signature matrix
    """

    random.seed(0)

    r = len(binary_vec)
    c = len(binary_vec[0])
    binary_vec = np.array(binary_vec)

    # Find k.
    k = nextprime(r - 1)

    # Generate n random hash functions.
    hash_params = np.empty((n, 2))
    for i in range(n):
        # Generate a, b, and k.
        a = random.randint(1, k - 1)
        b = random.randint(1, k - 1)
        hash_params[i, 0] = a
        hash_params[i, 1] = b

    # Initialize signature matrix to infinity for each element.
    signature = np.full((n, c), np.inf)

    # Loop through the binary vector representation matrix once, to compute the signature matrix.
    for row in range(1, r + 1):
        # Compute each of the n random hashes once for each row.
        e = np.ones(n)
        row_vec = np.full(n, row)
        x = np.stack((e, row_vec), axis=1)
        row_hash = np.sum(hash_params * x, axis=1) % k

        for i in range(n):
            # Update column j if and only if it contains a one and its current value is larger than the hash value for
            # the signature matrix row i.
            updates = np.where(binary_vec[row - 1] == 0, np.inf, row_hash[i])
            signature[i] = np.where(updates < signature[i], row_hash[i], signature[i])
    return signature


def common_binary(data):
    """
    Finds and reports the most common binary and numeric features.

    :param data: a list of items
    :return:
    """
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
