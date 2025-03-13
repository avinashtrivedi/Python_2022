import csv

import pandas
from surprise import Reader, Dataset


def load_ratings(nrows=None):
    reader = Reader(line_format='user item rating', sep=';', skip_lines=1, rating_scale=(0, 10))
    dataframe = pandas.read_csv('BX-Book-Ratings.csv', sep=';', encoding='ISO-8859-1', error_bad_lines=False, header=0,
                                nrows=nrows, dtype={'User-ID': 'string', 'ISBN': 'string', 'Book-Rating': 'float'})
    print(dataframe.info())
    data_set = Dataset.load_from_df(dataframe, reader=reader)

    isbn_to_title = {}
    with open('BX_Books.csv', newline='', encoding='ISO-8859-1') as csv_file:
        reader = csv.reader(csv_file, delimiter=';', quotechar='\"')
        next(reader)
        for row in reader:
            isbn = row[0]
            title = row[1]
            isbn_to_title[isbn] = title

    return data_set, isbn_to_title
