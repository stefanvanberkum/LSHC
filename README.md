# LSHC

Locality sensitive hashing with clustering (LSHC) for product duplicate detection.

## Installation

- Set-up a virtual environment using Python 3.9.
- Install the requirements by running the following command in your virtual environment: ```conda install --file requirements.txt``` (or ```pip install -r requirements.txt```).
- Install English spacy model by running the following command: ```python -m spacy download en_core_web_md```.
- You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/.

## What's included?

- data: folder containing the TV data used in our paper.
- results: folder containing the raw result of our code.
- LSH.py: module that provides LSH functionality.
- data_loader.py: module that provides data loading functionality for reading JSON files such as TVs.json.
- main.py: the main environment, this is where you need to be if you just want to run the code.

## How to run?

- Simply run the main method, and change any settings at the start of main.main().
- Note that running times may be long (a few hours in our case).
