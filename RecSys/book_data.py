import csv
import os
import sys
from collections import defaultdict

import pandas
from surprise import Dataset
from surprise import Reader


class BookData:
    isbn_to_title = {}
    title_to_isbn = {}
    ratings_data_filename = 'BX-Book-Ratings.csv'
    book_data_filename = 'BX_Books.csv'

    def load_data(self, nrows=None):
        ratings_data_set = 0
        self.isbn_to_title = {}
        self.title_to_isbn = {}

        reader = Reader(line_format='user item rating', sep=';', skip_lines=1, rating_scale=(0, 10))
        dataframe = pandas.read_csv(self.ratings_data_filename, sep=';', encoding='ISO-8859-1', error_bad_lines=False,
                                    header=0,
                                    nrows=nrows, dtype={'User-ID': 'string', 'ISBN': 'string', 'Book-Rating': 'float'})
        ratings_data_set = Dataset.load_from_df(dataframe, reader=reader)

        with open(self.book_data_filename, newline='', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file, delimiter=';', quotechar='\"')
            next(reader)
            for row in reader:
                isbn = row[0]
                title = row[1]
                self.isbn_to_title[isbn] = title
                self.title_to_isbn[title] = isbn

        return ratings_data_set

    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_data_filename, newline='', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file, delimiter=';', quotechar='\"')
            next(reader)
            for row in reader:
                isbn = row[1]
                ratings[isbn] += 1
        rank = 1
        for isbn, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[isbn] = rank
            rank += 1
        return rankings

    def get_book_title(self, isbn):
        if isbn in self.isbn_to_title:
            return self.isbn_to_title[isbn]
        else:
            return ""

    def get_isbn(self, title):
        if title in self.title_to_isbn:
            return self.title_to_isbn[title]
        else:
            return 0
