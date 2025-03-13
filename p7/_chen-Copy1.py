# + endofcell="--"
# # %load _chen.py
# # +
# # %%writefile _chen.py
#project:p7
#submitter: chen
#partner:none
#hours:8
#orginal
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np 

class UserPredictor:
    def __init__(self):
        self.model =  LogisticRegression()
        
    def fit(self, users, logs, y):
        m = self.helper(users,logs)
        self.model.fit(m[['past_purchase_amt','seconds']], y['y'])
        
    def predict(self, test_users, test_logs):
        m = self.helper(test_users, test_logs)
        y_pred = self.model.predict(m[['past_purchase_amt','seconds']])
        return y_pred
    
    def helper(self,users,logs):
        dic = dict()
        for user_id in list(logs['user_id']):
            if user_id in dic:
                dic[user_id] = dic[user_id]+1
            else:
                dic[user_id] = 0 
        f = logs.groupby('user_id').sum()

        users = users.set_index('user_id')
        users['seconds'] = f['seconds']
        me = users['seconds'].mean()
        users.fillna(me, inplace = True)
        l = []
        for w in users['badge']:
            if w == 'bronze':
                l.append(1)
            elif w == 'silver':
                l.append(2)
            else:#w == 'gold':
                l.append(3) 
        l2 = []
        for w in users['past_purchase_amt']:
            l2.append(w*w)
        users['Med']= l
        users['p2'] = l2
        return users
# -



# --


