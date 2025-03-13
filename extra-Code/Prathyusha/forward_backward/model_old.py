import pandas as pd
import numpy as np
from data_loading import salesforce_pipe_agg, utilization_trends, salesforce_wins, revenue, moving_avg_features, create_all_data, rev_to_pred, hours_charged
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from data_loading import revenue
from sklearn.feature_selection import SequentialFeatureSelector
from joblib import dump, load
import os
import shap

def load_df(model_t_plus = 1):

    df_sf = salesforce_pipe_agg()
    df_wins = salesforce_wins()
    #df_utilization, df_resource_count, df_mean_above, df_mean_below, df_count_above, df_count_below =  utilization_trends()
    df_utilization, df_resource_count, df_mean_dev =  utilization_trends()
    df_rev = revenue()
    df_hours = hours_charged()

    all_data = [df_sf, df_wins, df_utilization, df_mean_dev, df_resource_count, df_hours]

    all_data = moving_avg_features(all_data)

    all_data.append(df_rev)

    df_rev_to_pred = rev_to_pred(df_rev, t_step=model_t_plus)
    all_data.append(df_rev_to_pred)

    all_data_df = create_all_data(all_data)
    all_data_df = all_data_df[[x for x in all_data_df.columns if "Managing Director" not in x and "Partner" not in x and "Specialist" not in x and 'Intern' not in x and 'D+' not in x]]
    all_data_df = all_data_df[all_data_df['Identifier'].apply(lambda x: 'Unknown' not in x and 'None' not in x and 'Unassigned' not in x)]
    all_data_df = all_data_df[all_data_df['Identifier'].apply(lambda x: '7.0_2021.0' not in x and '7.0_2022.0' not in x )]

    return all_data_df

def impute_missing(all_data_df):

    imputer = KNNImputer()
    to_impute = all_data_df[list(set(all_data_df.columns) - set(['Growth Cell', 'Time', 'Identifier', "EM% mean + 1", "Revenue Sum + 1", "Revenue Sum", "EM% mean"]))]
    all_data_df[list(set(all_data_df.columns) - set(['Growth Cell', 'Time', 'Identifier', "EM% mean + 1", "Revenue Sum + 1", "Revenue Sum", "EM% mean"]))] = imputer.fit_transform(to_impute)

    return all_data_df

def preprocess_df(all_data_df, months_to_predict = ['6.0_2022.0'], model_t_plus = 1):

    bd = pd.read_csv(r"datafull\business_days.csv")
    bd["Time"] = bd.apply(lambda x: x["Month"].astype(float).astype(str) + "_"  + x["Year"].astype(float).astype(str) , axis = 1)
    bd_map = bd[['Time', 'Business Days']].set_index("Time").to_dict()["Business Days"]
    all_data_df["business_days"] = all_data_df['Time'].apply(lambda x: bd_map[x])
    all_data_df["Revenue Sum"] = all_data_df["Revenue Sum"]/all_data_df["business_days"]


    if model_t_plus == 1:
        bd['Business Days + 1'] = bd['Business Days'].shift(-1)
        bd_map_1 = bd[['Time', 'Business Days + 1']].set_index("Time").to_dict()["Business Days + 1"]
        all_data_df["business_days + 1"] = all_data_df['Time'].apply(lambda x: bd_map_1[x])
        all_data_df["Revenue Sum + 1"] = all_data_df["Revenue Sum + 1"]/all_data_df["business_days + 1"]
        all_data_df = all_data_df.drop('business_days + 1', axis = 1)
    else:
        bd['Business Days + 2'] = bd['Business Days'].shift(-2)
        bd_map_2 = bd[['Time', 'Business Days + 2']].set_index("Time").to_dict()["Business Days + 2"]
        all_data_df["business_days + 2"] = all_data_df['Time'].apply(lambda x: bd_map_2[x])
        all_data_df["Revenue Sum + 1"] = all_data_df["Revenue Sum + 1"]/all_data_df["business_days + 2"]
        all_data_df = all_data_df.drop('business_days + 2', axis = 1)

    
    '''all_data_df['Associate_Count_Below'] = all_data_df['Associate_Count_Below']*100/all_data_df['Associate_Count']
    all_data_df['Associate_Count_Above'] = all_data_df['Associate_Count_Above']*100/all_data_df['Associate_Count']
    all_data_df['Senior Associate_Count_Below'] = all_data_df['Senior Associate_Count_Below']*100/all_data_df['Senior Associate_Count']
    all_data_df['Senior Associate_Count_Above'] = all_data_df['Senior Associate_Count_Above']*100/all_data_df['Senior Associate_Count']
    all_data_df['Manager_Count_Below'] = all_data_df['Manager_Count_Below']*100/all_data_df['Manager_Count']
    all_data_df['Manager_Count_Above'] = all_data_df['Manager_Count_Above']*100/all_data_df['Manager_Count']
    all_data_df['Senior Manager_Count_Below'] = all_data_df['Senior Manager_Count_Below']*100/all_data_df['Senior Manager_Count']
    all_data_df['Senior Manager_Count_Above'] = all_data_df['Senior Manager_Count_Above']*100/all_data_df['Senior Manager_Count']
    all_data_df['Director_Count_Below'] = all_data_df['Director_Count_Below']*100/all_data_df['Director_Count']
    all_data_df['Director_Count_Above'] = all_data_df['Director_Count_Above']*100/all_data_df['Director_Count']'''
    
    
    all_data_df["team_size"] = all_data_df["Associate_Count"] + all_data_df["Senior Associate_Count"] + all_data_df["Manager_Count"] + all_data_df["Senior Manager_Count"] + all_data_df["Director_Count"]
    all_data_df["Associate_Count"] = all_data_df["Associate_Count"]*100/all_data_df["team_size"]
    all_data_df["Senior Associate_Count"] = all_data_df["Senior Associate_Count"]*100/all_data_df["team_size"]
    all_data_df["Manager_Count"] = all_data_df["Manager_Count"]*100/all_data_df["team_size"]
    all_data_df["Senior Manager_Count"] = all_data_df["Senior Manager_Count"]*100/all_data_df["team_size"]
    all_data_df["Director_Count"] = all_data_df["Director_Count"]*100/all_data_df["team_size"]
    
    
    all_data_df["Revenue Sum"] = np.log1p(all_data_df['Revenue Sum'])
    all_data_df["Revenue Sum + 1"] = np.log1p(all_data_df['Revenue Sum + 1'])
    all_data_df["team_size"] = np.log1p(all_data_df["team_size"])
    all_data_df['Amount (converted)_x'] = np.log1p(all_data_df['Amount (converted)_x'])
    all_data_df['Amount (converted)_y'] = np.log1p(all_data_df['Amount (converted)_y'])
    all_data_df['Opportunity Weighted Amount'] = np.log1p(all_data_df['Opportunity Weighted Amount'])

    all_data_df = all_data_df.rename(columns = {'Amount (converted)_x': 'Amount added in pipe', 'Opportunity ID_x': 'Entries added in pipe',
        'Ultimate Parent Account Name_x': 'Unique accounts added in pipe', 'Amount (converted)_y':'Amount won',
        'Opportunity ID_y':'Projects won', 'Ultimate Parent Account Name_y':'Unique accounts won', 'Opportunity Weighted Amount':'Opportunity Weighted Amount', 
        'Associate_Util':'Associate_Util', 'Director_Util':'Director_Util', 'Manager_Util':'Manager_Util', 'Senior Associate_Util':'Senior Associate_Util',
        'Senior Manager_Util':'Senior Manager_Util', 'Associate_Mean_Dev' : 'Associate diff from Target', 'Director_Mean_Dev': 'Director diff from Target',
        'Frac 4 Code M-':'Frac 4 Code M-', 'Frac 4 Code Manager-Director':'Frac 4 Code Manager-Director',
        'Frac PD Code M-':'Frac PD Code M-', 'Frac PD Code Manager-Director':'Frac PD Code Manager-Director',
        'Frac Client Code M-':'Frac Client Code M-', 'Frac Client Code Manager-Director':'Frac Client Code Manager-Director',
        'Senior Associate_Mean_Dev' : 'SA diff from Target', 'Senior Manager_Mean_Dev' : 'SM diff from Target', 'Manager_Mean_Dev':'Manager diff from Target',
        'Associate_Count': 'Fracn of team as As', 'Director_Count': 'Fracn of teams as Ds', 'Manager_Count': 'Fracn of teams as Ms',
        'Senior Associate_Count': 'Fracn of team as SAs', 'Senior Manager_Count': 'Fracn of teams as SMs', 'EM% mean + 1':'EM% mean + 1',
        'Revenue Sum + 1': 'Revenue Sum + 1', 'Revenue Sum': 'Revenue Sum', 'EM% mean': 'EM% mean', 'business_days':'business_days',
        'team_size': 'log of team size'})

    all_data_df['Growth Cell'] = all_data_df['Growth Cell'].astype('category')
    
    to_pred = all_data_df[all_data_df["Time"].apply(lambda x: x in months_to_predict) ]

    all_data_df = all_data_df[~all_data_df[['Revenue Sum','EM% mean','EM% mean + 1','Revenue Sum + 1']].apply(lambda x: any(x.isna()), axis = 1)]
    all_data_df = all_data_df[all_data_df["Time"].apply(lambda x: x not in months_to_predict) ]

    gc_time = all_data_df[['Growth Cell', 'Time']]
    all_data_df = all_data_df.drop(['Identifier', 'Time'], axis = 1)

    return all_data_df, to_pred, gc_time


'''def train_xgb(t, predict_for, keep_columns):
    
    all_data_df = load_df(model_t_plus=t)
    all_data_df = impute_missing(all_data_df)
    all_data_df, to_pred, gc_time = preprocess_df(all_data_df, months_to_predict=predict_for, model_t_plus=t)
    
    model_xgb = XGBRegressor(n_estimators = 50, max_depth = 4)
    
    y_log = all_data_df['Revenue Sum + 1']
    X = all_data_df[keep_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.20, random_state=42)
    model_xgb.fit(X_train, y_train)
    
    to_pred_gc_time = to_pred[['Growth Cell', 'Time']]
    X_copy = to_pred[keep_columns]
    pred_rev = (np.exp(model_xgb.predict(X_copy)) - 1)
    to_pred_gc_time['Revenue Sum'] = pred_rev
    
    
    
    output = {'train_acc': model_xgb.score(X_train, y_train),
             'test_acc' : model_xgb.score(X_test, y_test),
             'trained_model':model_xgb,
             'forecasts': to_pred_gc_time}
    
    return output'''


def train_lgb(t, predict_for, keep_columns):
    
    all_data_df = load_df(model_t_plus=t)
    all_data_df = impute_missing(all_data_df)
    all_data_df, to_pred, gc_time = preprocess_df(all_data_df, months_to_predict=predict_for, model_t_plus=t)
    
    model_lgb = lgb.LGBMRegressor()
    
    y_log = all_data_df['Revenue Sum + 1']
    X = all_data_df[keep_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.20, random_state=42)
    model_lgb.fit(X_train, y_train)
    
    to_pred_gc_time = to_pred[['Growth Cell', 'Time']]
    X_copy = to_pred[keep_columns]
    pred_rev = (np.exp(model_lgb.predict(X_copy)) - 1)
    to_pred_gc_time['Revenue Sum'] = pred_rev
    
    
    
    output = {'train_acc': model_lgb.score(X_train, y_train),
             'test_acc' : model_lgb.score(X_test, y_test),
             'trained_model':model_lgb,
             'forecasts': to_pred_gc_time}
    
    return output

def prepare_output(xgb_out, lgb_out, model_t, t):
    
    revenue_actuals = revenue()
    df = xgb_out['forecasts']
    df['Revenue Forecast'] = (xgb_out['forecasts']['Revenue Sum'] + xgb_out['forecasts']['Revenue Sum'])/2
    df['Year'] = df['Time'].apply(lambda x: int(float(x.split('_')[1])))
    df['Month'] = df['Time'].apply(lambda x: int(float(x.split('_')[0]))+t)
    df['model'] = model_t
    df = df[['Growth Cell', 'Year', 'Month', 'Revenue Forecast', 'model']]
    final_out = revenue_actuals.merge(df, how = 'outer', left_on=['Growth Cell', 'Year', 'Month'], right_on = ['Growth Cell', 'Year', 'Month'])
    final_out = final_out[(final_out['Year'] == 2022) & (final_out['Month'].apply(lambda x: x in list(range(5,7+t))))]
    final_out = final_out.sort_values(['Growth Cell', 'Year', 'Month'])
    
    
    return final_out[['Growth Cell', 'Year', 'Month','Revenue Sum','Revenue Forecast', 'model']]

def get_feats(sfs, feature_set):
    feats = []
    for i,val in enumerate(sfs.get_support()):
        if val:
            feats.append(feature_set[i])
    return feats

def prepare_output_xgb(xgb_out, model_t, t):
    
    revenue_actuals = revenue()
    df = xgb_out['forecasts']
    df['Revenue Forecast'] = xgb_out['forecasts']['Revenue Sum']
    df['Year'] = df['Time'].apply(lambda x: int(float(x.split('_')[1])))
    df['Month'] = df['Time'].apply(lambda x: int(float(x.split('_')[0]))+t)
    df['model'] = model_t
    df = df[['Growth Cell', 'Year', 'Month', 'Revenue Forecast', 'model']]
    final_out = revenue_actuals.merge(df, how = 'outer', left_on=['Growth Cell', 'Year', 'Month'], right_on = ['Growth Cell', 'Year', 'Month'])
    final_out = final_out[(final_out['Year'] == 2022) & (final_out['Month'].apply(lambda x: x in list(range(5,7+t))))]
    final_out = final_out.sort_values(['Growth Cell', 'Year', 'Month'])
    
    
    return final_out[['Growth Cell', 'Year', 'Month','Revenue Sum','Revenue Forecast', 'model']]


def train_xgb(t, predict_for, feature_set, model_folder):
    
    all_data_df = load_df(model_t_plus=t)
    all_data_df = impute_missing(all_data_df)
    all_data_df, to_pred, gc_time = preprocess_df(all_data_df, months_to_predict=predict_for, model_t_plus=t)
    
    for i in range(1, len(feature_set)):
#         print(os.path.join(os.path.join(model_folder,str(i)),'_sfs_forward.joblib'))
        if os.path.exists(os.path.join(model_folder,f"{str(i)}_sfs_forward.joblib")):
            pass
        else:
            print('Model Not saved')
            xgb = XGBRegressor(n_estimators = 50, max_depth = 4, n_jobs =-1,tree_method="hist", enable_categorical=True,random_state=42)
            sfs = SequentialFeatureSelector(xgb,n_features_to_select=i,n_jobs=-1)
            sfs.fit(all_data_df[feature_set], all_data_df['Revenue Sum + 1'])
            dump(sfs, model_folder + str(i) + '_sfs_forward.joblib')

            xgb = XGBRegressor(n_estimators = 50, max_depth = 4, n_jobs =-1, tree_method="hist", enable_categorical=True,random_state=42)
            sfs = SequentialFeatureSelector(xgb,n_features_to_select=i,direction='backward',n_jobs=-1)
            sfs.fit(all_data_df[feature_set], all_data_df['Revenue Sum + 1'])
            dump(sfs, model_folder + str(i) + '_sfs_backward.joblib')
    
    stats = []
    for i in range(1,len(feature_set)):
        sfs = load(os.path.join(model_folder,f"{str(i)}_sfs_forward.joblib"))
        X = sfs.transform(all_data_df[feature_set])
        y_log = all_data_df['Revenue Sum + 1']
        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.30, random_state=42)
        if os.path.exists(os.path.join(model_folder,f"{str(i)}_xgb_forward.joblib")):
            xgb = load(os.path.join(model_folder,f"{str(i)}_xgb_forward.joblib"))
        else:
            
            xgb = XGBRegressor(n_estimators = 50,  n_jobs =-1,max_depth = 4, tree_method="hist", enable_categorical=True)
            xgb.fit(X_train, y_train)
            dump(xgb, model_folder + str(i) + '_xgb_forward.joblib')
        stats.append([get_feats(sfs, feature_set), xgb.score(X_train, y_train), xgb.score(X_test, y_test), 'forward'])

        sfs = load(os.path.join(model_folder,f"{str(i)}_sfs_backward.joblib"))
        X = sfs.transform(all_data_df[feature_set])
        y_log = all_data_df['Revenue Sum + 1']
        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.30, random_state=42)
        if os.path.exists(os.path.join(model_folder,f"{str(i)}_xgb_backward.joblib")):
            xgb = load(os.path.join(model_folder,f"{str(i)}_xgb_backward.joblib"))
        else:
            xgb = XGBRegressor(n_estimators = 50,  n_jobs =-1,max_depth = 4, tree_method="hist", enable_categorical=True)
            xgb.fit(X_train, y_train)
            dump(xgb, model_folder + str(i) + '_xgb_backward.joblib')
        stats.append([get_feats(sfs, feature_set), xgb.score(X_train, y_train), xgb.score(X_test, y_test), 'backward'])

    stats.sort(key=lambda x: x[2], reverse=True)
    
    final_features = list(stats[0][0])
    X = all_data_df[final_features]
    y_log = all_data_df['Revenue Sum + 1']
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.30, random_state=42)
    final_model = load(os.path.join(model_folder,str(len(stats[0][0])) + '_xgb_' + stats[0][-1] + '.joblib'))
    
    to_pred_gc_time = to_pred[['Growth Cell', 'Time']]
    X_copy = to_pred[final_features]
    pred_rev = (np.exp(final_model.predict(X_copy)) - 1)
    to_pred_gc_time['Revenue Sum'] = pred_rev
    
    output = {'train_acc': final_model.score(X_train, y_train),
             'test_acc' : final_model.score(X_test, y_test),
             'trained_model':final_model,
             'forecasts': to_pred_gc_time,
             'accuracy_stats':stats}
    return output


def combine_outputs(t_1_model, t_2_model, scope_gcs):
    
    t_1_preds = prepare_output_xgb(t_1_model, 't+1',1)
    t_1_preds['Revenue Sum'] = t_1_preds.apply(lambda x: x['Revenue Sum']/20 if x['Month'] == 5 else (x['Revenue Sum']/22 if x['Month'] == 6 else x['Revenue Sum']), axis = 1)
    t_1_preds['diff'] = t_1_preds["Revenue Forecast"] - t_1_preds['Revenue Sum'].shift()
    t_1_preds['diff'] = t_1_preds['diff'].apply(lambda x: 1 if x<0 else (0 if x>0 else x))
    t_2_preds = prepare_output_xgb(t_2_model, 't+2',2)
    t_2_preds['Revenue Sum'] = t_2_preds.apply(lambda x: x['Revenue Sum']/20 if x['Month'] == 5 else (x['Revenue Sum']/22 if x['Month'] == 6 else x['Revenue Sum']), axis = 1)
    t_2_preds['diff'] = t_2_preds["Revenue Forecast"] - t_2_preds['Revenue Sum'].shift()
    t_2_preds['diff2'] = t_2_preds["Revenue Forecast"] - t_2_preds['Revenue Sum'].shift(2)
    t_2_preds['diff_fin'] = t_2_preds.apply(lambda x: 1 if x['diff']<0 else( 0 if x['diff']>0 else( 1 if x['diff2'] < 0 else(0 if x['diff2']>0 else x['diff'])) )   , axis = 1)
    t_2_preds['ticker_2'] = t_2_preds['diff_fin']
    t_1_preds['ticker_1'] = t_1_preds['diff']
    
    explainer_xgb1 = shap.TreeExplainer(t_1_model['trained_model'])
    explainer_xgb2 = shap.TreeExplainer(t_2_model['trained_model'])

    t = 1
    predict_for = ['6.0_2022.0']
    all_data_df = load_df(model_t_plus=t)
    all_data_df = impute_missing(all_data_df)
    all_data_df, to_pred_1, gc_time = preprocess_df(all_data_df, months_to_predict=predict_for, model_t_plus=t)

    t = 2
    predict_for = ['6.0_2022.0', '5.0_2022.0']
    all_data_df_ = load_df(model_t_plus=t)
    all_data_df_ = impute_missing(all_data_df_)
    all_data_df_, to_pred_2, gc_time_ = preprocess_df(all_data_df_, months_to_predict=predict_for, model_t_plus=t)

    shap_values_xgb1 = explainer_xgb1.shap_values(to_pred_1[t_1_model['accuracy_stats'][0][0]])

    shap_values_xgb2 = explainer_xgb2.shap_values(to_pred_2[t_2_model['accuracy_stats'][0][0]])

    p = [[(i,x) for i,x in enumerate(k)] for k in  shap_values_xgb1]

    [t.sort(key = lambda x: x[1]) for t in p]

    bottom_3 = [[t_1_model['accuracy_stats'][0][0][i[0]] for i in t][:3] for t in p]

    top_3 = [[t_1_model['accuracy_stats'][0][0][i[0]] for i in t][-3:] for t in p]

    [x.reverse() for x in top_3]

    explainers_xgb1 = to_pred_1[['Growth Cell', 'Time']]

    explainers_xgb1['top_3'] = top_3

    explainers_xgb1['bottom_3'] = bottom_3

    explainers_xgb1['Month'] = explainers_xgb1['Time'].apply(lambda x: int(float(x.split('_')[0]))+1)
    explainers_xgb1['Year'] = explainers_xgb1['Time'].apply(lambda x: int(float(x.split('_')[1])))

    explainers_xgb1['top_3'] = explainers_xgb1['top_3'].apply(lambda x: (', ').join([k for k in x if k!='log of team size']))
    explainers_xgb1['bottom_3'] = explainers_xgb1['bottom_3'].apply(lambda x:  (', ').join([k for k in x if k!='log of team size']))

    t_1_preds_ = t_1_preds.merge(explainers_xgb1[['Growth Cell', 'Year', 'Month', 'top_3', 'bottom_3']], how = 'outer', on = ['Growth Cell', 'Year', 'Month'])

    p = [[(i,x) for i,x in enumerate(k)] for k in  shap_values_xgb2]

    [t.sort(key = lambda x: x[1]) for t in p]

    bottom_3 = [[t_2_model['accuracy_stats'][0][0][i[0]] for i in t][:3] for t in p]

    top_3 = [[t_2_model['accuracy_stats'][0][0][i[0]] for i in t][-3:] for t in p]

    [x.reverse() for x in top_3]

    explainers_xgb2 = to_pred_2[['Growth Cell', 'Time']]

    explainers_xgb2['top_3'] = top_3

    explainers_xgb2['bottom_3'] = bottom_3

    explainers_xgb2['Month'] = explainers_xgb2['Time'].apply(lambda x: int(float(x.split('_')[0]))+2)
    explainers_xgb2['Year'] = explainers_xgb2['Time'].apply(lambda x: int(float(x.split('_')[1])))

    explainers_xgb2['top_3'] = explainers_xgb2['top_3'].apply(lambda x: (', ').join([k for k in x if k!='log of team size']))
    explainers_xgb2['bottom_3'] = explainers_xgb2['bottom_3'].apply(lambda x:  (', ').join([k for k in x if k!='log of team size']))

    t_2_preds_ = t_2_preds.merge(explainers_xgb2[['Growth Cell', 'Year', 'Month', 'top_3', 'bottom_3']], how = 'outer',  on = ['Growth Cell', 'Year', 'Month'])

    t_2_preds_['variables_explain'] = t_2_preds_.apply(lambda x: x['bottom_3'] if x['ticker_2'] ==1 else ( x['top_3'] if x['model'] == 't+2' else x['model'] ) , axis = 1)

    t_1_preds_['variables_explain'] = t_1_preds_.apply(lambda x: x['bottom_3'] if x['ticker_1'] ==1 else ( x['top_3'] if x['model'] == 't+1' else x['model'] ) , axis = 1)

    to_save = t_1_preds_[['Growth Cell', 'Year', 'Month', 'Revenue Sum', 'Revenue Forecast',
           'model', 'variables_explain', 'ticker_1']].append(t_2_preds_[['Growth Cell', 'Year', 'Month', 'Revenue Sum', 'Revenue Forecast',
           'model', 'variables_explain', 'ticker_2']]).sort_values(['Growth Cell', 'Year', 'Month']).drop_duplicates()

    to_save['ticker'] = to_save.apply(lambda x: x['ticker_1'] if not np.isnan(x['ticker_1']) else ( x['ticker_2'] if not np.isnan(x['ticker_2']) else x['ticker_1'] ), axis = 1)

    to_save['Scope_GCs'] = to_save['Growth Cell'].apply(lambda x: 1 if x in scope_gcs else 0)
    
    return to_save
