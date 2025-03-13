# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:48:02 2018

@author: Frank
"""
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline


class EvaluationData:

    def __init__(self, data, popularity_rankings):
        self.rankings = popularity_rankings

        # Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test

        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

        # Compute similarity matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)

    def get_full_train_set(self):
        return self.fullTrainSet

    def get_full_anti_test_set(self):
        return self.fullAntiTestSet

    def get_anti_test_set_for_user(self, testSubject):
        train_set = self.fullTrainSet
        fill = train_set.global_mean
        anti_test_set = []
        u = train_set.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in train_set.ur[u]])
        anti_test_set += [(train_set.to_raw_uid(u), train_set.to_raw_iid(i), fill) for
                          i in train_set.all_items() if
                          i not in user_items]
        return anti_test_set

    def get_train_set(self):
        return self.trainSet

    def get_test_set(self):
        return self.testSet

    def get_LOOCV_train_set(self):
        return self.LOOCVTrain

    def get_LOOCV_test_set(self):
        return self.LOOCVTest

    def get_LOOCV_anti_test_set(self):
        return self.LOOCVAntiTestSet

    def get_similarities(self):
        return self.simsAlgo

    def get_popularity_rankings(self):
        return self.rankings
