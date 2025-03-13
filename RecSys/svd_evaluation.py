# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from book_data import BookData
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from evaluator import Evaluator

import random
import numpy as np


def load_book_data():
    book_data = BookData()
    print("Loading book ratings...")
    data = book_data.load_data()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    # rankings = book_data.get_popularity_ranks()
    rankings = []
    return book_data, data, rankings


np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
book_data, evaluationData, rankings = load_book_data()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(book_data, '276725')
