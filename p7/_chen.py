#project:p7
#submitter: chen
#partner:none
#hours:8
#orginal
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np 

class UserPredictor:
    def __init__(self):
         self.model = Pipeline([
                                ("poly", PolynomialFeatures(degree=2)),
                                ("std", StandardScaler()),
                                ("lr", LogisticRegression(max_iter=200)),
                                ])

#         self.model =  LogisticRegression()
        
    def fit(self, train_users, train_logs, y):
        helper_df = self.helper(train_users,train_logs)
        self.model.fit(helper_df, y['y'])
        
    def predict(self, test_users, test_logs):
        helper_df = self.helper(test_users, test_logs)
        y_pred = self.model.predict(helper_df)
        return y_pred
    
    def helper(self,users_data,logs_data):
        
        logs_seconds_by_id = logs_data.groupby('user_id').sum()

        users_data.set_index('user_id',inplace=True)
        users_data['seconds'] = logs_seconds_by_id['seconds']
        avg = users_data['seconds'].mean()
        users_data.fillna(avg, inplace = True)
        users_data = pd.get_dummies(users_data,columns=['badge'])
        users_data.drop(['names'],inplace=True,axis=1)
        return users_data
