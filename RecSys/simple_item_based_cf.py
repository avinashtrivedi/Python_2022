import heapq
from collections import defaultdict
from operator import itemgetter

from surprise import KNNBasic

from load_data import load_ratings

dataset, isbn_to_title = load_ratings(10000)

train_set = dataset.build_full_trainset()

sim_matrix = KNNBasic(sim_options={
    'name': 'cosine',
    'user_based': False
}).fit(train_set).compute_similarities()

test_subject = '276725'
k = 10

test_subject_iid = train_set.to_inner_uid(test_subject)

test_user_ratings = train_set.ur[test_subject_iid]

kn = heapq.nlargest(k, test_user_ratings, key=lambda t: t[1])

# kn = []
# for rating in test_user_ratings:
#     if rating[1] > 7.0:
#         kn.append(rating)

candidates = defaultdict(float)

for item_id, rating in kn:
    sim_row = sim_matrix[item_id]
    for inner_id, score in enumerate(sim_row):
        candidates[inner_id] += score * (rating / 10.0)

books_read = {}

for item_id, rating in train_set.ur[test_subject_iid]:
    books_read[item_id] = 1

recommendations = []

position = 0
for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if itemID not in books_read:
        isbn = train_set.to_raw_iid(itemID)
        if isbn not in isbn_to_title:
            continue
        recommendations.append(f'{isbn_to_title[isbn]}, ISBN = {isbn}')
        position += 1
        if position > 10:
            break

for r in recommendations:
    print("Book: ", r)
