import heapq
from collections import defaultdict
from operator import itemgetter

from surprise import KNNBasic

from load_data import load_ratings

dataset, isbn_to_title = load_ratings(100000)

train_set = dataset.build_full_trainset()

sim_matrix = KNNBasic(sim_options={
    'name': 'cosine',
    'user_based': True
}).fit(train_set).compute_similarities()

test_subject = '276729'
k = 10

test_subject_iid = train_set.to_inner_uid(test_subject)

similarity_row = sim_matrix[test_subject_iid]

print(similarity_row)

similar_users = []

for inner_id, rating in enumerate(similarity_row):
    if inner_id != test_subject_iid:
        similar_users.append((inner_id, rating))

# kn = heapq.nlargest(k, similar_users, key=lambda t: t[1])

kn = []
for rating in similar_users:
    if rating[1] > 0:
        print(rating)
    if rating[1] > 0.95:
        kn.append(rating)

print(len(kn))

candidates = defaultdict(float)

for user in kn:
    inner_id = user[0]
    user_sim_score = user[1]
    user_ratings = train_set.ur[inner_id]
    for rating in user_ratings:
        candidates[rating[0]] += (rating[1] / 10.0) * user_sim_score

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
