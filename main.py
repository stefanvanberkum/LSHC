"""
This is the main execution environment for the MSMP++ procedure.

https://github.com/stefanvanberkum/MSMP
"""

from data_loader import load
from LSH import common_binary, lsh, lsh_old

def main():
    """
    Runs the whole MSMP++ procedure, and stores results.

    :return:
    """

    identify_common_binary = False
    run_msm = True

    file_path = "data/TVs.json"

    data_list, data_dict = load(file_path)

    if identify_common_binary:
        common_binary(data_list)

    if run_msm:
        lsh(data_list)
        lsh_old(data_list)
        print()


if __name__ == '__main__':
    main()
