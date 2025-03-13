import pandas as pd

import numpy as np

from data_loading import salesforce_pipeline, utilization_trends, salesforce_wins, revenue, moving_avg_features, create_all_data, rev_to_pred
from sklearn.impute import KNNImputer

def load_df(model_t_plus = 1):

    df_sf = salesforce_pipeline()
    df_wins = salesforce_wins()
    df_utilization, df_resource_count, df_mean_above, df_mean_below, df_count_above, df_count_below =  utilization_trends()
    df_rev = revenue()

    all_data = [df_sf, df_wins, df_utilization, df_count_above, df_count_below, df_mean_above, df_mean_below, df_resource_count]

    all_data = moving_avg_features(all_data)

    all_data.append(df_rev)

    df_rev_to_pred = rev_to_pred(df_rev, t_step=model_t_plus)
    all_data.append(df_rev_to_pred)

    all_data_df = create_all_data(all_data)
    all_data_df = all_data_df[[x for x in all_data_df.columns if "Managing Director" not in x and "Partner" not in x and "Specialist" not in x and 'Intern' not in x]]
    all_data_df = all_data_df[all_data_df['Identifier'].apply(lambda x: 'Unknown' not in x and 'None' not in x and 'Unassigned' not in x)]
    all_data_df = all_data_df[all_data_df['Identifier'].apply(lambda x: '7.0_2021.0' not in x and '7.0_2022.0' not in x )]

    return all_data_df

def impute_missing(all_data_df):

    imputer = KNNImputer()
    to_impute = all_data_df[list(set(all_data_df.columns) - set(['Growth Cell', 'Time', 'Identifier', "EM% mean + 1", "Revenue Sum + 1", "Revenue Sum", "EM% mean"]))]
    all_data_df[list(set(all_data_df.columns) - set(['Growth Cell', 'Time', 'Identifier', "EM% mean + 1", "Revenue Sum + 1", "Revenue Sum", "EM% mean"]))] = imputer.fit_transform(to_impute)

    return all_data_df

def preprocess_df(all_data_df, months_to_predict = ['6.0_2022.0'], model_t_plus = 1):

    # bd = pd.read_csv("/Users/svatsavai001/Documents/TransformationForecasting/business_days.csv")
    bd = pd.read_csv(r"D:\OneDrive - NITT\CODE\Python Scripts\Code_2022\forward_backward_feature_Selection\business_days.csv")
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

    
    all_data_df['Associate_Count_Below'] = all_data_df['Associate_Count_Below']*100/all_data_df['Associate_Count']
    all_data_df['Associate_Count_Above'] = all_data_df['Associate_Count_Above']*100/all_data_df['Associate_Count']
    all_data_df['Senior Associate_Count_Below'] = all_data_df['Senior Associate_Count_Below']*100/all_data_df['Senior Associate_Count']
    all_data_df['Senior Associate_Count_Above'] = all_data_df['Senior Associate_Count_Above']*100/all_data_df['Senior Associate_Count']
    all_data_df['Manager_Count_Below'] = all_data_df['Manager_Count_Below']*100/all_data_df['Manager_Count']
    all_data_df['Manager_Count_Above'] = all_data_df['Manager_Count_Above']*100/all_data_df['Manager_Count']
    all_data_df['Senior Manager_Count_Below'] = all_data_df['Senior Manager_Count_Below']*100/all_data_df['Senior Manager_Count']
    all_data_df['Senior Manager_Count_Above'] = all_data_df['Senior Manager_Count_Above']*100/all_data_df['Senior Manager_Count']
    all_data_df['Director_Count_Below'] = all_data_df['Director_Count_Below']*100/all_data_df['Director_Count']
    all_data_df['Director_Count_Above'] = all_data_df['Director_Count_Above']*100/all_data_df['Director_Count']
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

    all_data_df = all_data_df.rename(columns = {'Amount (converted)_x': 'Amount added in pipe', 'Opportunity ID_x': 'Entries added in pipe',
        'Ultimate Parent Account Name_x': 'Unique accounts added in pipe', 'Amount (converted)_y':'Amount won',
        'Opportunity ID_y':'Projects won', 'Ultimate Parent Account Name_y':'Unique accounts won', 'Associate_Util':'Associate_Util',
        'Director_Util':'Director_Util', 'Manager_Util':'Manager_Util', 'Senior Associate_Util':'Senior Associate_Util',
        'Senior Manager_Util':'Senior Manager_Util', 'Associate_Count_Above': 'Num As Above Target', 'Director_Count_Above':'Num Ds Above Target',
        'Manager_Count_Above': 'Num Ms above target', 'Senior Associate_Count_Above': 'Num SAs above target',
        'Senior Manager_Count_Above': 'Num SMs above target', 'Associate_Count_Below':'Num As Below Target',
        'Director_Count_Below': 'Num Ds Below Target', 'Manager_Count_Below':'Num Ms below target',
        'Senior Associate_Count_Below': 'Num SAs below target', 'Senior Manager_Count_Below':'Num SMs below target',
        'Associate_Mean_Above': 'As overutilized by', 'Director_Mean_Above':'Ds overutilized by', 'Manager_Mean_Above': 'Ms overutilized by',
        'Senior Associate_Mean_Above':'SAs overutilized by', 'Senior Manager_Mean_Above':'SMs overutilized by',
        'Associate_Mean_Below': 'As underutilized by', 'Director_Mean_Below':'Ds underutilized by', 'Manager_Mean_Below': 'Ms underutilized by',
        'Senior Associate_Mean_Below':'SAs underutilized by', 'Senior Manager_Mean_Below': 'SMs underutilized by',
        'Associate_Count': 'Fracn of team as As', 'Director_Count': 'Fracn of teams as Ds', 'Manager_Count': 'Fracn of teams as Ms',
        'Senior Associate_Count': 'Fracn of team as SAs', 'Senior Manager_Count': 'Fracn of teams as SMs', 'EM% mean + 1':'EM% mean + 1',
        'Revenue Sum + 1': 'Revenue Sum + 1', 'Revenue Sum': 'Revenue Sum', 'EM% mean': 'EM% mean', 'business_days':'business_days',
        'team_size': 'log of team size'})

    all_data_df['Growth Cell'] = all_data_df['Growth Cell'].astype('category')
    
    to_pred = all_data_df[all_data_df["Time"].apply(lambda x: x in months_to_predict) ]

    all_data_df = all_data_df[~all_data_df[['Revenue Sum','EM% mean','EM% mean + 1','Revenue Sum + 1']].apply(lambda x: any(x.isna()), axis = 1)]
    gc_time = all_data_df[['Growth Cell', 'Time']]
    all_data_df = all_data_df.drop(['Identifier', 'Time'], axis = 1)

    return all_data_df, to_pred, gc_time


