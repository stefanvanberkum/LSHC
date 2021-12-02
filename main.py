"""
This is the main execution environment for the MSMP++ procedure.

https://github.com/stefanvanberkum/MSMP
"""

from data_loader import load
from LSH import lsh

def main():
    """
    Runs the whole MSMP++ procedure, and stores results.

    :return:
    """

    file_path = "data/TVs.json"

    data_list, data_dict = load(file_path)
    lsh(data_list)
    print()


if __name__ == '__main__':
    main()
