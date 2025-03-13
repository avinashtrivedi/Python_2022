import pandas as pd

full_data=pd.read_csv('owid-covid-data_2.csv')

data=full_data[(full_data['location']=='Bangladesh')|(full_data['location']=='India')|(full_data['location']=='China')]

data.to_csv('input.csv',index=False)