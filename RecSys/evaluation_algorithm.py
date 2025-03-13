# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:45:33 2018

@author: Frank
"""
from recommender_metrics import RecommenderMetrics


class EvaluatedAlgorithm:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.get_train_set())
        predictions = self.algorithm.test(evaluationData.get_test_set())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if doTopN:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.get_LOOCV_train_set())
            leftOutPredictions = self.algorithm.test(evaluationData.get_LOOCV_test_set())
            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.get_LOOCV_anti_test_set())
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.get_full_train_set())
            allPredictions = self.algorithm.test(evaluationData.get_full_anti_test_set())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(topNPredicted,
                                                                  evaluationData.get_full_train_set().n_users,
                                                                  ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.get_similarities())

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted,
                                                            evaluationData.get_popularity_rankings())

        if verbose:
            print("Analysis complete.")

        return metrics

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm
