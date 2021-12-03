"""
This is the main execution environment for the MSMP++ procedure.

https://github.com/stefanvanberkum/MSMP
"""

from data_loader import load
from LSH import common_binary, lsh, lsh_old
import time

def main():
    """
    Runs the whole MSMP++ procedure, and stores results.

    :return:
    """

    identify_common_binary = False
    run_msm = True

    file_path = "data/TVs.json"

    start_time = time.time()

    data_list, data_dict = load(file_path)

    if identify_common_binary:
        common_binary(data_list)

    if run_msm:
        lsh(data_list)
        lsh_old(data_list)

    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")


if __name__ == '__main__':
    main()
