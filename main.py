"""
This is the main execution environment for the LSHC procedure.

https://github.com/stefanvanberkum/LSHC
"""

import random
import re
import time
from math import comb

import numpy as np
import spacy
from sklearn.cluster import KMeans

from LSH import convert_binary, convert_binary_alt, convert_binary_old, minhash, lsh, common_count
from data_loader import load


def main():
    """
    Runs the whole LSHC procedure, and stores results in a csv file.

    :return:
    """

    identify_common_count = False
    run_lsh = True
    write_result = True

    thresholds = [x / 100 for x in range(5, 100, 5)]
    bootstraps = 5
    random.seed(0)

    file_path = "data/TVs.json"
    result_path = "results/"

    start_time = time.time()

    data_list, duplicates = load(file_path)

    if identify_common_count:
        common_count(data_list)

    if run_lsh:
        if write_result:
            with open(result_path + "results.csv", 'w') as out:
                out.write(
                    "t,comparisons,pq,pc,f1_star,f1,comparisons_alt,pq_alt,pc_alt,f1_star_alt,f1_alt,comparisons_old,"
                    "pq_old,pc_old,f1_star_old,f1_old\n")

        for t in thresholds:
            print("t = ", t)

            # Initialize statistics, where results = [comparisons, pq, pc, f1_star, f1].
            results = np.zeros(5)
            results_alt = np.zeros(5)
            results_old = np.zeros(5)

            for run in range(bootstraps):
                data_sample, duplicates_sample = bootstrap(data_list, duplicates)
                comparisons_run, pq_run, pc_run, f1_star_run, f1_run = do_lshc(data_sample, duplicates_sample, t)
                results += np.array([comparisons_run, pq_run, pc_run, f1_star_run, f1_run])
                comparisons_alt_run, pq_alt_run, pc_alt_run, f1_star_alt_run, f1_alt_run = do_lshc_alt(data_sample,
                                                                                                       duplicates_sample,
                                                                                                       t)
                results_alt += np.array([comparisons_alt_run, pq_alt_run, pc_alt_run, f1_star_alt_run, f1_alt_run])
                comparisons_old_run, pq_old_run, pc_old_run, f1_star_old_run, f1_old_run = do_lshc_old(data_sample,
                                                                                                       duplicates_sample,
                                                                                                       t)
                results_old += np.array([comparisons_old_run, pq_old_run, pc_old_run, f1_star_old_run, f1_old_run])

            # Compute average statistics over all bootstraps.
            statistics = results / bootstraps
            statistics_alt = results_alt / bootstraps
            statistics_old = results_old / bootstraps

            if write_result:
                with open(result_path + "results.csv", 'a') as out:
                    out.write(str(t))
                    for stat in statistics:
                        out.write("," + str(stat))
                    for stat in statistics_alt:
                        out.write("," + str(stat))
                    for stat in statistics_old:
                        out.write("," + str(stat))
                    out.write("\n")

    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")


def do_lshc(data_list, duplicates, t):
    """
    Bins items using MinHash and LSH, clusters using K-means, and computes and returns performance metrics based on
    the matrix of true duplicates.

    :param data_list: a list of items
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :param t: the threshold value
    :return: the fraction of comparisons, pair quality, pair completeness, F_1^* measure, and F_1 measure.
    """

    binary_vec = convert_binary(data_list)
    n = round(round(0.5 * len(binary_vec)) / 100) * 100
    signature = minhash(binary_vec, n)
    candidates = lsh(signature, t)

    # Compute number of comparisons.
    comparisons = np.sum(candidates) / 2
    comparison_frac = comparisons / comb(len(data_list), 2)

    # Compute matrix of correctly binned duplicates, where element (i, j) is equal to one if item i and item j are
    # duplicates, and correctly classified as such by LSH.
    correct = np.where(duplicates + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    # Compute Pair Quality (PQ)
    pq = n_correct / comparisons

    # Compute Pair Completeness (PC)
    pc = n_correct / (np.sum(duplicates) / 2)

    # Compute F_1^* measure.
    f1_star = 2 * pq * pc / (pq + pc)

    # Cluster and compute F_1 measure.
    tp, precision = cluster(data_list, candidates, duplicates)
    recall = tp / (np.sum(duplicates) / 2)
    f1 = 2 * precision * recall / (precision + recall)

    return comparison_frac, pq, pc, f1_star, f1


def do_lshc_alt(data_list, duplicates, t):
    """
    Bins items using MinHash and LSH, clusters using K-means, and computes and returns performance metrics based on
    the matrix of true duplicates.

    :param data_list: a list of items
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :param t: the threshold value
    :return: the fraction of comparisons, pair quality, pair completeness, F_1^* measure, and F_1 measure
    """

    binary_vec = convert_binary_alt(data_list)
    n = round(round(0.5 * len(binary_vec)) / 100) * 100
    signature = minhash(binary_vec, n)
    candidates = lsh(signature, t)

    # Compute number of comparisons.
    comparisons = np.sum(candidates) / 2
    comparison_frac = comparisons / comb(len(data_list), 2)

    # Compute matrix of correctly binned duplicates, where element (i, j) is equal to one if item i and item j are
    # duplicates, and correctly classified as such by LSH.
    correct = np.where(duplicates + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    # Compute Pair Quality (PQ)
    pq = n_correct / comparisons

    # Compute Pair Completeness (PC)
    pc = n_correct / (np.sum(duplicates) / 2)

    # Compute F_1^* measure.
    f1_star = 2 * pq * pc / (pq + pc)

    # Cluster and compute F_1 measure.
    tp, precision = cluster(data_list, candidates, duplicates)
    recall = tp / (np.sum(duplicates) / 2)
    f1 = 2 * precision * recall / (precision + recall)

    return comparison_frac, pq, pc, f1_star, f1


def do_lshc_old(data_list, duplicates, t):
    """
    Bins items using MinHash and LSH, clusters using K-means, and computes and returns performance metrics based on
    the matrix of true duplicates.

    NOTE. This is the old implementation by Hartveld et al. (2018), implemented for evaluation purposes.

    :param data_list: a list of items
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :param t: the threshold value
    :return: the fraction of comparisons, pair quality, pair completeness, F_1^* measure, and F_1 measure
    """

    binary_vec = convert_binary_old(data_list)
    n = round(round(0.5 * len(binary_vec)) / 100) * 100
    signature = minhash(binary_vec, n)
    candidates = lsh(signature, t)

    # Compute number of comparisons.
    comparisons = np.sum(candidates) / 2
    comparison_frac = comparisons / comb(len(data_list), 2)

    # Compute matrix of correctly binned duplicates, where element (i, j) is equal to one if item i and item j are
    # duplicates, and correctly classified as such by LSH.
    correct = np.where(duplicates + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    # Compute Pair Quality (PQ)
    pq = n_correct / comparisons

    # Compute Pair Completeness (PC)
    pc = n_correct / (np.sum(duplicates) / 2)

    # Compute F_1 measure.
    f1_star = 2 * pq * pc / (pq + pc)

    # Cluster and compute F_1 measure.
    tp, precision = cluster(data_list, candidates, duplicates)
    recall = tp / (np.sum(duplicates) / 2)
    f1 = 2 * precision * recall / (precision + recall)

    return comparison_frac, pq, pc, f1_star, f1


def bootstrap(data_list, duplicates):
    """
    Creates a bootstrap by sampling n elements from the data with replacement, where n denotes the size of the original
    dataset.

    :param data_list: a list of data
    :param duplicates: a binary matrix where item (i, j) is equal to one if items i and j are duplicates, and zero
    otherwise
    :return: a bootstrap sample of the data and the corresponding duplicate matrix
    """

    # Compute indices to be included in the bootstrap.
    indices = [random.randint(x, len(data_list) - 1) for x in [0] * len(data_list)]

    # Collect samples.
    data_sample = [data_list[index] for index in indices]
    duplicates_sample = np.take(np.take(duplicates, indices, axis=0), indices, axis=1)
    return data_sample, duplicates_sample


def cluster(data_list, candidates, duplicates):
    """
    Clusters the candidate pairs using the pairwise title similarity and K-means clustering.

    :param data_list: the list of all items
    :param candidates: the binary matrix of candidate pairs, where element (i, j) is equal to one if item i and j are a
    candidate pair.
    :param duplicates: the binary matrix of duplicates, where element (i, j) is equal to one if item i and j are a
    candidate pair.
    :return: the number of true positives and the precision
    """

    nlp = spacy.load("en_core_web_md", disable=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat'])

    titles = []
    duplicate_vec = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if candidates[i, j] == 1:
                # Find model words in the titles.
                mw_i = re.findall(
                    "(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.["
                    "0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))",
                    data_list[i]["title"])
                mw_title_i = ""
                for match in mw_i:
                    if mw_i[0] != '':
                        mw_title_i += match[0]
                    else:
                        mw_title_i += match[1]
                mw_j = re.findall(
                    "(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.["
                    "0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))",
                    data_list[j]["title"])
                mw_title_j = ""
                for match in mw_j:
                    if mw_j[0] != '':
                        mw_title_j += match[0]
                    else:
                        mw_title_j += match[1]
                titles.append([mw_title_i, mw_title_j])

                if duplicates[i, j] == 1:
                    # (i, j) true duplicates.
                    duplicate_vec.append(1)
                else:
                    # (i, j) not duplicates.
                    duplicate_vec.append(0)

    titles = np.array(titles)

    def similarity(x):
        """
        Computes similarity of two items using spaCy.
        :param x: list of two indices
        :return: the similarity of these items
        """
        title_i = nlp(str(x[0]))
        title_j = nlp(str(x[1]))
        return title_i.similarity(title_j)

    # Compute similarities of candidate pair product titles using spaCy.
    similarities = np.apply_along_axis(similarity, 1, titles)

    duplicate_vec = np.array(duplicate_vec)

    # Cluster using K-means.
    kmeans = KMeans(n_clusters=2, random_state=random.randint(1, 100000)).fit(similarities.reshape(-1, 1))
    labels = kmeans.labels_
    mean_0 = np.sum(np.where(labels == 0, similarities, 0)) / np.count_nonzero(labels == 0)
    mean_1 = np.sum(np.where(labels == 1, similarities, 0)) / np.count_nonzero(labels == 1)

    # Check which pairs are more similar on average, and declare these to be the duplicates.
    predictions = np.where(labels == 0, 1, 0) if mean_0 > mean_1 else np.where(labels == 1, 1, 0)

    # Compute precision.
    tp = np.sum(np.where(predictions == duplicate_vec, predictions, 0))
    precision = tp / np.sum(predictions)

    return tp, precision


if __name__ == '__main__':
    main()
