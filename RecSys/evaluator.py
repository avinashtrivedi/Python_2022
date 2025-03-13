from evaluation_algorithm import EvaluatedAlgorithm
from evaluation_data import EvaluationData


class Evaluator:
    algorithms = []

    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed

    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.get_name(), "...")
            results[algorithm.get_name()] = algorithm.evaluate(self.dataset, doTopN)

        # Print results
        print("\n")

        if doTopN:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if doTopN:
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print(
                "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def SampleTopNRecs(self, book_data, testSubject, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.get_name())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.get_full_train_set()
            algo.get_algorithm().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.dataset.get_anti_test_set_for_user(testSubject)

            predictions = algo.get_algorithm().test(testSet)

            recommendations = []

            print("\nWe recommend:")
            for user_id, isbn, actualRating, estimatedRating, _ in predictions:
                recommendations.append((isbn, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            counter = 0
            for ratings in recommendations:
                if counter > k:
                    break

                title = book_data.get_book_title(ratings[0])
                if title == '':
                    continue

                print(title, ratings[1])
                counter += 1
