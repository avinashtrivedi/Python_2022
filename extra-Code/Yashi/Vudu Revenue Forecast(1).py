# -*- coding: utf-8 -*-
"""

@author: Divya.Veeramani
"""

# Import Required Packages
import pandas as pd
import numpy as np
import os
import pandas.io.sql as sqlio
import psycopg2
from prophet import Prophet
import datetime as dt
import warnings
from sqlalchemy import create_engine
from pandas.io.sql import SQLTable
warnings.filterwarnings("ignore")

# Establish redshift connection
user_name = '*****'
password = '*****'
db_name = '*****'
server = '*****'
driver = '*****'
port = '*****'
conn = psycopg2.connect(dbname=db_name, host=server, port=port, user=user_name,password=password)

# Change current directory
os.chdir('D:/Fandango/Weekly forecast Vudu')

# Set Maximum Date for forecast
max_date = '2022-12-31'

# Get Daily Level Data
file1 = open("Vudu_Forecast_data_daily.sql","rt")
QUERY1  = file1.read()
trans_data = sqlio.read_sql_query(QUERY1, conn)
cols = ['Transaction Date', 'Est Revenue', 'Vod Revenue', 'Est Trannx',
       'Vod Trannx', 'Nr Est Rev', 'Nr Vod Rev', 'Lib Est Rev', 'Lib Vod Rev',
       'D2d Revenue', 'Total Rev', 'Discounted Rev', 'New Paids', 'Ex Paids']
trans_data.columns = cols
trans_data['Transaction Date'] = pd.to_datetime(trans_data['Transaction Date'])
date_range = pd.period_range(min(trans_data['Transaction Date']),max_date)
date_range = pd.DataFrame(date_range)
date_range.columns = ['Transaction Date2']
date_range['Transaction Date2'] = pd.to_datetime(date_range['Transaction Date2'].astype(str))
#trans_data = pd.concat([trans_data,date_range], ignore_index=True, axis=1)
trans_data = pd.merge(trans_data,date_range, left_on = 'Transaction Date', right_on = 'Transaction Date2', how = 'right')
trans_data = trans_data.iloc[:,1:]
temp_cols = list(trans_data.columns)
temp_cols = [temp_cols[-1]] + temp_cols[:-1]
trans_data = trans_data[temp_cols]
trans_data.columns = cols
trans_data = trans_data.sort_values(by = ['Transaction Date'])
trans_data = trans_data.reset_index()

# Get Weekly Level Data
file1 = open("Vudu_Forecast_data_weekly.sql","rt")
QUERY1  = file1.read()
trans_data_weekly = sqlio.read_sql_query(QUERY1, conn)
trans_data_weekly.columns = cols
trans_data_weekly['Transaction Date'] = pd.to_datetime(trans_data_weekly['Transaction Date'])
date_range = pd.period_range(min(trans_data_weekly['Transaction Date']),max_date,freq = 'W')
date_range = pd.DataFrame(date_range)
date_range[0] = date_range[0].astype(str).str.split("/", n=1, expand = True)[0]
date_range[0] = pd.to_datetime(date_range[0])
date_range.columns = ['Transaction Date2']
date_range['Transaction Date2'] = pd.to_datetime(date_range['Transaction Date2'].astype(str))
#trans_data = pd.concat([trans_data,date_range], ignore_index=True, axis=1)
trans_data_weekly = pd.merge(trans_data_weekly,date_range, left_on = 'Transaction Date', right_on = 'Transaction Date2', how = 'right')
trans_data_weekly = trans_data_weekly.iloc[:,1:]
temp_cols = list(trans_data_weekly.columns)
temp_cols = [temp_cols[-1]] + temp_cols[:-1]
trans_data_weekly = trans_data_weekly[temp_cols]
trans_data_weekly.columns = cols
trans_data_weekly = trans_data_weekly.sort_values(by = ['Transaction Date'])
trans_data_weekly = trans_data_weekly.reset_index()

# Get Monthly Level Data
file1 = open("Vudu_Forecast_data_monthly.sql","rt")
QUERY1  = file1.read()
trans_data_monthly = sqlio.read_sql_query(QUERY1, conn)
trans_data_monthly.columns = cols
trans_data_monthly['Transaction Date'] = pd.to_datetime(trans_data_monthly['Transaction Date'])
date_range = pd.period_range(min(trans_data_monthly['Transaction Date']),max_date,freq = 'M')
date_range = pd.DataFrame(date_range)
date_range.columns = ['Transaction Date2']
date_range['Transaction Date2'] = pd.to_datetime(date_range['Transaction Date2'].astype(str))
#trans_data = pd.concat([trans_data,date_range], ignore_index=True, axis=1)
trans_data_monthly = pd.merge(trans_data_monthly,date_range, left_on = 'Transaction Date', right_on = 'Transaction Date2', how = 'right')
trans_data_monthly = trans_data_monthly.iloc[:,1:]
temp_cols = list(trans_data_monthly.columns)
temp_cols = [temp_cols[-1]] + temp_cols[:-1]
trans_data_monthly = trans_data_monthly[temp_cols]
trans_data_monthly.columns = cols
trans_data_monthly = trans_data_monthly.sort_values(by = ['Transaction Date'])
trans_data_monthly = trans_data_monthly.reset_index()

# Give train cut off
now = dt.datetime.now()
dow = now.weekday()
train_cut_off = now - dt.timedelta(days = dow+7)
train_cut_off = train_cut_off.strftime("%Y-%m-%d")
# train_cut_off = '2021-10-04'
print(train_cut_off)

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

trans_data['Transaction Date'] = trans_data['Transaction Date'].astype(str)
trans_data['Transaction Date'] = pd.to_datetime(trans_data['Transaction Date']).dt.strftime('%Y-%m-%d')

# Remove First half of 2020
trans_data = trans_data.loc[(trans_data['Transaction Date']< '2020-01-01')| (trans_data['Transaction Date']> '2020-06-30'),:].reset_index()
trans_data['week_day'] = pd.DatetimeIndex(trans_data['Transaction Date']).dayofweek

#---------------------------EX PAIDS DAILY FORECAST--------------------------
# Train and Test Split
train = trans_data.loc[(trans_data['Transaction Date']< train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
test = trans_data.loc[(trans_data['Transaction Date']>= train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
train.rename(columns = {'Transaction Date' : 'ds', 'Ex Paids' : 'y'}, inplace = True)

# Train prophet - Added weekly and yearly seasonality, US holidays
m = Prophet(weekly_seasonality = True, yearly_seasonality = True)
m.add_country_holidays(country_name='US')
m.fit(train)

# Create future data frame with seasonality variables
future = m.make_future_dataframe(periods=test.shape[0])

# Predict and Plot
forecast = m.predict(future)
fig = m.plot_components(forecast)
forecast['week_day'] = trans_data['week_day']
forecast['yhat'] = forecast['yhat'] * 0.90 #1.05#1.02
forecast['yhat_lower'] = forecast['yhat_lower'] * 0.90
forecast['yhat_upper'] = forecast['yhat_upper'] * 0.90

# Get Accuracy metrics
print(forecast_accuracy(forecast.loc[forecast['ds']>=train_cut_off,'yhat'], test['Ex Paids']))
ex_paids_pred = forecast.loc[forecast['ds']>=train_cut_off,['ds','yhat','yhat_lower','yhat_upper']]
ex_paids_pred['Ex Paids'] = test['Ex Paids']
ex_paids_pred['week_day'] = test['week_day']
ex_paids_pred.head()

# Get predicted Ex Paids and concatenate data for Total Revenue prediction
# Taken Actual value of Ex Paids till July 2021 after which predicted values are taken 
test.loc[test['Transaction Date']>=train_cut_off,'Ex Paids'] = forecast.loc[forecast['ds']>=train_cut_off,'yhat']
train.rename(columns = {'y':'Ex Paids','ds':'Transaction Date'}, inplace = True)
print(test.head())
print(train.head())
trans_data = pd.concat([train[['Transaction Date','Total Rev','Ex Paids']],test[['Transaction Date','Total Rev','Ex Paids']]])
trans_data['year'] = pd.to_datetime(trans_data['Transaction Date']).dt.year
trans_data['year'].value_counts()

#---------------------------EX PAIDS WEEKLY FORECAST--------------------------
trans_data_weekly['week_day'] = pd.DatetimeIndex(trans_data_weekly['Transaction Date']).dayofweek

# Train and Test Split
train = trans_data_weekly.loc[(trans_data_weekly['Transaction Date']< train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
test = trans_data_weekly.loc[(trans_data_weekly['Transaction Date']>= train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
train.rename(columns = {'Transaction Date' : 'ds', 'Ex Paids' : 'y'}, inplace = True)

# Train prophet - Added weekly and yearly seasonality, US holidays
m = Prophet(yearly_seasonality = True)
m.add_country_holidays(country_name='US')
m.fit(train)

# Create future data frame with seasonality variables
future = m.make_future_dataframe(freq='W',periods=test.shape[0])
print(future.tail())

# Predict and Plot
forecast = m.predict(future)
fig = m.plot_components(forecast)
forecast['week_day'] = trans_data_weekly['week_day']
forecast['ds'] = forecast['ds']+dt.timedelta(days = 1) 
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast['yhat'] = forecast['yhat'] * 0.95 # 1.12
forecast['yhat_lower'] = forecast['yhat_lower'] * 0.95
forecast['yhat_upper'] = forecast['yhat_upper'] * 0.95

# Get Accuracy metrics
print(forecast_accuracy(forecast.loc[forecast['ds']>=train_cut_off,'yhat'], test['Ex Paids']))
ex_paids_week = forecast.loc[forecast['ds']>=train_cut_off,['ds','yhat','yhat_lower','yhat_upper']]
ex_paids_week['Ex Paids'] = test['Ex Paids']
ex_paids_week['week_day'] = test['week_day']
ex_paids_week['Ex Paids'].iloc[1] = 0
print(ex_paids_week.head())

#---------------------------EX PAIDS MONTHLY FORECAST--------------------------
trans_data_monthly['Transaction Date'] = trans_data_monthly['Transaction Date'].astype(str)
trans_data_monthly['Transaction Date'] = pd.to_datetime(trans_data_monthly['Transaction Date'])
trans_data_monthly['week_day'] = pd.DatetimeIndex(trans_data_monthly['Transaction Date']).dayofweek

train_cut_off_month = (dt.datetime.now() - dt.timedelta(days = 7)).replace(day=1).strftime("%Y-%m-%d")

# Train and Test Split
train = trans_data_monthly.loc[(trans_data_monthly['Transaction Date']< train_cut_off_month),['Transaction Date','Total Rev','Ex Paids']]
test = trans_data_monthly.loc[(trans_data_monthly['Transaction Date']>= train_cut_off_month),['Transaction Date','Total Rev','Ex Paids']]
train.rename(columns = {'Transaction Date' : 'ds', 'Ex Paids' : 'y'}, inplace = True)

# Train prophet - Added weekly and yearly seasonality, US holidays
m = Prophet(yearly_seasonality = True)
m.add_country_holidays(country_name='US')
m.fit(train)

# Create future data frame with seasonality variables
future = m.make_future_dataframe(freq='M',periods=test.shape[0])
future.loc[future['ds'].dt.day>1,'ds'] = future.loc[future['ds'].dt.day>1,'ds'] + dt.timedelta(days = 1)
print(future.tail())

# Predict and Plot
forecast = m.predict(future)
fig = m.plot_components(forecast)
#forecast['week_day'] = trans_data_monthly['week_day']
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast['yhat'] = forecast['yhat'] * 0.95 #1.12
forecast['yhat_lower'] = forecast['yhat_lower'] * 0.95
forecast['yhat_upper'] = forecast['yhat_upper'] * 0.95

# Get Accuracy metrics
print(forecast_accuracy(forecast.loc[forecast['ds']>=train_cut_off_month,'yhat'], test['Ex Paids']))
ex_paids_month = forecast.loc[forecast['ds']>=train_cut_off_month,['ds','yhat','yhat_lower','yhat_upper']]
ex_paids_month['Ex Paids'] = test['Ex Paids']
ex_paids_month.reset_index(inplace = True, drop = True)

first_month = ex_paids_month['ds'].iloc[0].month
curr_month = dt.date.today().month

if first_month != curr_month:
    ex_paids_month = ex_paids_month.drop([0], axis=0).reset_index(drop = True)
    
print(ex_paids_month.head())

# Ex Paids Output
ex_paids_pred['% error'] = pd.Series(ex_paids_pred.yhat)/pd.Series(ex_paids_pred['Ex Paids'])-1
ex_paids_pred["mape"] = np.abs(ex_paids_pred["% error"])

ex_paids_week['% error'] = pd.Series(ex_paids_week.yhat)/pd.Series(ex_paids_week['Ex Paids'])-1
ex_paids_week["mape"] = np.abs(ex_paids_week["% error"])

ex_paids_month['% error'] = pd.Series(ex_paids_month.yhat)/pd.Series(ex_paids_month['Ex Paids'])-1
ex_paids_month["mape"] = np.abs(ex_paids_month["% error"])


#---------------------------TOTAL REVENUE FORECAST--------------------------
# Prophet
trans_data['week_day'] = pd.DatetimeIndex(trans_data['Transaction Date']).dayofweek
# Train and Test Split
train = trans_data.loc[(trans_data['Transaction Date']< train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
test = trans_data.loc[(trans_data['Transaction Date']>= train_cut_off),['Transaction Date','Total Rev','Ex Paids','week_day']]
train.rename(columns = {'Transaction Date' : 'ds', 'Total Rev' : 'y'}, inplace = True)

# Train Prophet - Added only Ex Paids regressor
m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m.add_regressor('Ex Paids')
m.add_country_holidays(country_name='US')
m.fit(train)

# Prepare future data frame for prediction
future = m.make_future_dataframe(periods=test.shape[0])
future['Ex Paids'] = trans_data['Ex Paids']
future.tail()
future.fillna(0, inplace = True)

# Predict and Plot
forecast = m.predict(future)
fig = m.plot_components(forecast)

# Get Accuracy metrics
print(forecast_accuracy(forecast.loc[forecast['ds']>=train_cut_off,'yhat'], test['Total Rev']))
total_rev_pred = forecast.loc[forecast['ds']>=train_cut_off,['ds','yhat','yhat_lower','yhat_upper']]
total_rev_pred['Total Rev'] = test['Total Rev']
total_rev_pred['week_day'] = test['week_day']
total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat'] = total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat'] * 0.89 #1.04
total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat_lower'] = total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat_lower'] * 0.89
total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat_upper'] = total_rev_pred.loc[total_rev_pred['week_day']<=4,'yhat_upper'] * 0.89

total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat'] = total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat'] * 0.97 #1.06
total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat_lower'] = total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat_lower'] * 0.97
total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat_upper'] = total_rev_pred.loc[total_rev_pred['week_day']>4,'yhat_upper'] * 0.97

# Total Revenue Output
total_rev_pred['ds'] = pd.to_datetime(total_rev_pred['ds'])
total_rev_pred['WeekDate'] = total_rev_pred.apply(lambda row: row['ds'] - dt.timedelta(days=row['ds'].weekday()), axis=1)
total_rev_pred['Year Month'] = total_rev_pred['ds'].dt.to_period('M')
total_rev_pred['% error'] = pd.Series(total_rev_pred.yhat)/pd.Series(total_rev_pred['Total Rev'])-1
total_rev_pred["mape"] = np.abs(total_rev_pred["% error"])

total_rev_week = total_rev_pred.groupby(by = ['WeekDate'])['Total Rev','yhat','yhat_lower','yhat_upper'].agg({'Total Rev' : 'sum','yhat':'sum','yhat_lower':'sum', 'yhat_upper':'sum'})
total_rev_week.reset_index(inplace = True)
total_rev_week = total_rev_week.loc[total_rev_week['WeekDate'] >= train_cut_off,:]
total_rev_week['Total Rev'].iloc[1] = 0
total_rev_week['% error'] = pd.Series(total_rev_week.yhat)/pd.Series(total_rev_week['Total Rev'])-1
total_rev_week["mape"] = np.abs(total_rev_week["% error"])


total_rev_month = total_rev_pred.groupby(by = ['Year Month'])['Total Rev','yhat','yhat_lower','yhat_upper'].agg({'Total Rev' : 'sum','yhat':'sum','yhat_lower':'sum', 'yhat_upper':'sum'})
total_rev_month.reset_index(inplace = True)
total_rev_month = total_rev_month.loc[total_rev_month['Year Month'] >= train_cut_off,:]
total_rev_month['% error'] = pd.Series(total_rev_month.yhat)/pd.Series(total_rev_month['Total Rev'])-1
total_rev_month["mape"] = np.abs(total_rev_month["% error"])

first_month = total_rev_month['Year Month'].iloc[0].month
curr_month = dt.date.today().month

next_month = dt.datetime.now().replace(day=28) + dt.timedelta(days=4)
last_day = int((next_month - dt.timedelta(days=next_month.day)).day)

forecasted_days = last_day - (int(pd.to_datetime(train_cut_off).day) + 1)


if first_month != curr_month:
    total_rev_month = total_rev_month.drop([0], axis=0).reset_index(drop = True)
else:
    total_rev_month['yhat'].iloc[0] = (total_rev_month['yhat'].iloc[0]/forecasted_days) * last_day
    total_rev_month['yhat_lower'].iloc[0] = (total_rev_month['yhat_lower'].iloc[0]/forecasted_days) * last_day
    total_rev_month['yhat_upper'].iloc[0] = (total_rev_month['yhat_upper'].iloc[0]/forecasted_days) * last_day

# Get the required columns
ex_paids_pred = ex_paids_pred.loc[:,['ds','yhat','yhat_lower','yhat_upper','Ex Paids','mape']]
ex_paids_week = ex_paids_week.loc[:,['ds','yhat','yhat_lower','yhat_upper','Ex Paids','mape']]
ex_paids_month = ex_paids_month.loc[:,['ds','yhat','yhat_lower','yhat_upper','Ex Paids','mape']]
total_rev_pred = total_rev_pred.loc[:,['ds','yhat','yhat_lower','yhat_upper','Total Rev','mape']]
total_rev_week = total_rev_week.loc[:,['WeekDate','yhat','yhat_lower','yhat_upper','Total Rev','mape']]
total_rev_month = total_rev_month.loc[:,['Year Month','yhat','yhat_lower','yhat_upper','Total Rev','mape']]

# Rename columns
cols = ['Date','Prediction','Prediction Lower','Prediction Upper','Actual','MAPE']
ex_paids_pred.columns = cols
ex_paids_week.columns = cols
ex_paids_month.columns = cols
total_rev_pred.columns = cols
total_rev_week.columns = cols
total_rev_month.columns = cols

# Remove inf and nan
ex_paids_pred = ex_paids_pred.replace([np.nan,np.inf,0],'', regex = True)
ex_paids_week = ex_paids_week.replace([np.nan,np.inf,0],'', regex = True)
ex_paids_month = ex_paids_month.replace([np.nan,np.inf,0],'', regex = True)

total_rev_pred = total_rev_pred.replace([np.nan,np.inf,0],'', regex = True)
total_rev_week = total_rev_week.replace([np.nan,np.inf,0],'', regex = True)
total_rev_month = total_rev_month.replace([np.nan,np.inf,0],'', regex = True)

# Output files for Table upload
total_rev_month['Date'] = pd.to_datetime(total_rev_month['Date'].astype(str))
weekly_output = pd.merge(ex_paids_week,total_rev_week,on = 'Date', how = 'left')
monthly_output = pd.merge(ex_paids_month,total_rev_month,on = 'Date', how = 'left')

#weekly_output.drop(['Actual_x','MAPE_x','Actual_y','MAPE_y'], axis=1, inplace = True)
weekly_output.columns = ['Week_Starting_Date','Ex_Paids_Prediction','Ex_Paids_Pred_Low','Ex_Paids_Pred_High', 'Ex_Paids_Actual','Ex_Paids_Mape','Total_Revenue_Prediction','Total_Revenue_Pred_Low','Total_Revenue_Pred_High','Total_Revenue_Actual','Total_Revenue_Mape']
monthly_output.columns = ['Month_Starting_Date','Ex_Paids_Prediction','Ex_Paids_Pred_Low','Ex_Paids_Pred_High', 'Ex_Paids_Actual','Ex_Paids_Mape','Total_Revenue_Prediction','Total_Revenue_Pred_Low','Total_Revenue_Pred_High','Total_Revenue_Actual','Total_Revenue_Mape']

weekly_output['Refresh_Date'] = pd.to_datetime(dt.date.today())
monthly_output['Refresh_Date'] = pd.to_datetime(dt.date.today())

ex_paids_month.drop(['Actual','MAPE'], inplace = True, axis = 1)
total_rev_month.drop(['Actual','MAPE'], inplace = True, axis = 1)

#%%
# Export output Files
now = dt.datetime.now().strftime("%Y-%m-%d")
filename = "Vudu Forecast_"+str(now)+".xlsx"
writer = pd.ExcelWriter(filename)
ex_paids_pred.to_excel(writer,'Ex Paids Daily Prediction', index = False)
ex_paids_week.to_excel(writer,'Ex Paids Weekly Prediction', index = False)
ex_paids_month.to_excel(writer,'Ex Paids Monthly Prediction', index = False)
total_rev_pred.to_excel(writer,'Tot Rev Daily Prediction', index = False)
total_rev_week.to_excel(writer,'Tot Rev Weekly Prediction', index = False)
total_rev_month.to_excel(writer,'Tot Rev Monthly Prediction', index = False)
writer.save()

filename = "Weekly_Output_"+str(now)+".xlsx"
weekly_output.to_excel(filename, index = False)
filename = "Monthly_Output_"+str(now)+".xlsx"
monthly_output.to_excel(filename, index = False)
#%%


#%%

#-------------------------------------------------------------------------
# Upload Forecasts to Redshift tables
# Upload weekly prediction output
# Insert Records into table
def _execute_insert(self, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))
SQLTable._execute_insert = _execute_insert
conn = create_engine('postgresql://' + user_name + ':' + password + '@' + server + ':' + port + '/' + db_name)

# give the table name alone below.. schema should be given separately afterward
import time
START_COLUMN = 0
END_COLUMN = len(weekly_output)
BATCH_LIMIT = 500

for x in range(START_COLUMN, END_COLUMN, BATCH_LIMIT):
    print("Batch : " , x)
    start_time = time.time()
    data_df_sub = weekly_output.iloc[x: x + BATCH_LIMIT]
    data_df_sub.to_sql('lv_weekly_revenue_forecast',conn, schema = 'analytics_manual',index = False, if_exists = 'append',chunksize = BATCH_LIMIT)
    print("Completed", x, len(data_df_sub))
    print("--- %s seconds ---" % (time.time() - start_time))

# Upload monthly prediction output
# Insert Records into table
def _execute_insert(self, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(self.table.insert().values(data))
SQLTable._execute_insert = _execute_insert
conn = create_engine('postgresql://' + user_name + ':' + password + '@' + server + ':' + port + '/' + db_name)

# give the table name alone below.. schema should be given separately afterward
import time
START_COLUMN = 0
END_COLUMN = len(monthly_output)
BATCH_LIMIT = 500

for x in range(START_COLUMN, END_COLUMN, BATCH_LIMIT):
    print("Batch : " , x)
    start_time = time.time()
    data_df_sub = monthly_output.iloc[x: x + BATCH_LIMIT]
    data_df_sub.to_sql('lv_monthly_revenue_forecast',conn, schema = 'analytics_manual',index = False, if_exists = 'append',chunksize = BATCH_LIMIT)
    print("Completed", x, len(data_df_sub))
    print("--- %s seconds ---" % (time.time() - start_time))