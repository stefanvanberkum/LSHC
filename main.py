"""
This is the main execution environment for the MSMP++ procedure.

https://github.com/stefanvanberkum/MSMP
"""

from data_loader import load

def main():
    """
    Runs the whole MSMP++ procedure, and stores results.

    :return:
    """

    file_path = "data/TVs.json"

    data = load(file_path)
    print("")


if __name__ == '__main__':
    main()
