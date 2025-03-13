
import pandas as pd
import datetime
import os
import numpy as np

sai_dir = r'D:\OneDrive - NITT\CODE\Python Scripts\Code_2022\forward_backward_feature_Selection'

# pooja_dir = '/Users/pchopra018/Downloads/'
# waqar_dir = '/Users/waqarsarguroh/Downloads/'
# sai_dir = '/Users/svatsavai001/Downloads/'


'''def partner_names_gc(path = sai_dir + "PMD Perf Data as of 04.30.csv"):

    partner_perf = pd.read_csv(path)
    partner_perf = partner_perf[partner_perf['Engagement Partner'].apply(lambda x: "," in x)]
    partner_perf['Name'] = partner_perf['Engagement Partner'].apply(lambda x: x.split(', ')[1] + ' ' + x.split(', ')[0]) # change to last name plus first name

    name_gc_map = map_name_growth_cell(sai_dir + 'WoW Hours - All Staff.csv')

    partner_perf['gc'] = partner_perf['Name'].apply(lambda x: name_gc_map[x] if x in name_gc_map.keys() else 'Cannot Find')
    partner_perf['first_last'] = partner_perf['Name'].apply(lambda x: x.split()[0] +' '+ x.split()[-1]) # switch to first plus last name
    partner_perf['gc_2'] = partner_perf['first_last'].apply(lambda x: name_gc_map[x] if x in name_gc_map.keys() else 'Cannot Find')
    partner_perf['gc_3'] = partner_perf.apply(lambda x: x['gc'] if x['gc']!= 'Cannot Find' else x['gc_2'] , axis = 1) # Do OR between first+last / last+first
    partner_perf['gc_4'] = partner_perf.apply(lambda x: name_gc_map[[key for key in name_gc_map.keys() if x['first_last'].split()[1] in key][0]] if len([key for key in name_gc_map.keys() if x['first_last'].split()[1] in key]) == 1 else x['gc_3'], axis = 1) # If last name is unique
    partner_perf['Growth Cell'] = partner_perf.apply(lambda x: name_gc_map[[key for key in name_gc_map.keys() if x['first_last'].split()[0] in key][0]] if len([key for key in name_gc_map.keys() if x['first_last'].split()[0] in key]) == 1 else x['gc_4'], axis = 1) # If first name is unique
    partner_perf = partner_perf.drop(['gc', 'gc_2', 'gc_3', 'gc_4'], axis = 1)

    return partner_perf'''

def map_name_growth_cell_(path, exceptions = os.path.join(sai_dir, "P_MD_Constants/exceptions.csv")):
# def map_name_growth_cell_(path, exceptions = "P_MD_Constants/exceptions.csv"):
    wd_info = pd.read_csv(path)
    wd_info = wd_info[~wd_info['Last Name, First Name'].isna()]
    name_gc_map = wd_info[['Last Name, First Name', 'Growth Cell']].set_index('Last Name, First Name').to_dict()['Growth Cell']
    excepts = pd.read_csv(exceptions)
    for i, x in excepts.iterrows():
        if x['Workday Name'] in name_gc_map.keys():
            name_gc_map[x['Performance DB Name']] = name_gc_map[x['Workday Name']]

    return name_gc_map

def map_name_growth_cell(path = os.path.join(sai_dir,'WoW Hours - All Staff.csv')):

    wd_info = pd.read_csv(path)
    wd_info = wd_info[~wd_info['Name'].isna()]
    name_gc_map = wd_info[['Name', 'Growth Cell']].set_index('Name').to_dict()['Growth Cell']

    return name_gc_map

def map_empid_growth_cell(path = os.path.join(sai_dir,"WoW Hours - All Staff.csv")):

    wd_info = pd.read_csv(path)
    empid_gc_map = wd_info[['Employee ID', 'Growth Cell']].set_index('Employee ID').to_dict()['Growth Cell']

    return empid_gc_map

def salesforce_wins(path = os.path.join(sai_dir,"Wins Exports")):

    all_dfs_pipe = {}
    for file in os.listdir(path=path):
        if 'csv' in file:
            k = pd.read_csv(os.path.join(path, file), encoding = 'latin1')
            if k['Created Date'].apply(lambda x: int(x.split("/")[0])).max()>12 :
                k['Created Date'] = pd.to_datetime(k['Created Date'], dayfirst = True)
            else:
                k['Created Date'] = pd.to_datetime(k['Created Date'], dayfirst = False)
            if k['Close Date'].apply(lambda x: int(x.split("/")[0])).max()>12 :
                k['Close Date'] = pd.to_datetime(k['Close Date'], dayfirst = True)
            else:
                k['Close Date'] = pd.to_datetime(k['Close Date'], dayfirst = False)
            all_dfs_pipe[file] = k
    all_dfs = []
    for file, df in all_dfs_pipe.items():
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    all_dfs = all_dfs.drop_duplicates('Opportunity ID', keep = 'last')
    wins = all_dfs.reset_index(drop=True)

    empid_gc_map = map_empid_growth_cell()
    wins['Growth Cell'] = wins['Opportunity Lead: Employee Number'].apply(lambda x: empid_gc_map[x] if x in empid_gc_map.keys() else 'Cannot Find')
    wins['Created_date_year'] = wins['Created Date'].apply(lambda x: x.year)
    wins['Created_date_month'] = wins['Created Date'].apply(lambda x: x.month)

    wins_agg = wins[wins['Created Date']> datetime.datetime(2021, 7, 1)].groupby(['Growth Cell', 'Created_date_year', 'Created_date_month']).agg({'Amount (converted)': 'sum', 'Opportunity ID':'count', 'Ultimate Parent Account Name':pd.Series.nunique}) .reset_index()
    for gc in wins_agg['Growth Cell'].unique():
        k = wins_agg[wins_agg['Growth Cell'] == gc]
        for i,row in wins_agg[['Created_date_year','Created_date_month']].drop_duplicates().iterrows():
            if len(k[(k['Created_date_year'] == row['Created_date_year']) & (k['Created_date_month'] == row['Created_date_month'])])<1:
                wins_agg = wins_agg.append({'Growth Cell':gc, 'Created_date_year': row['Created_date_year'], 'Created_date_month': row['Created_date_month'], 'Amount (converted)':0, 'Opportunity ID': 0, 'Ultimate Parent Account Name': 0}, ignore_index = True)
    wins_agg = wins_agg.sort_values(['Growth Cell', 'Created_date_year', 'Created_date_month'])
    wins_agg = wins_agg.rename(columns = {'Created_date_year':'Year', 'Created_date_month':'Month'})

    return wins_agg

def salesforce_pipeline(path = os.path.join(sai_dir,'Daily Transformation Pipe Exports')):

    all_dfs_pipe = {}
    for file in os.listdir(path=path):
        if 'csv' in file:
            k = pd.read_csv(os.path.join(path, file), encoding = 'latin1')
            if k['Created Date'].apply(lambda x: int(x.split("/")[0])).max()>12 :
                k['Created Date'] = pd.to_datetime(k['Created Date'], dayfirst = True)
            else:
                k['Created Date'] = pd.to_datetime(k['Created Date'], dayfirst = False)
            if k['Close Date'].apply(lambda x: int(x.split("/")[0])).max()>12 :
                k['Close Date'] = pd.to_datetime(k['Close Date'], dayfirst = True)
            else:
                k['Close Date'] = pd.to_datetime(k['Close Date'], dayfirst = False)
            all_dfs_pipe[file] = k
    all_dfs = []
    for file, df in all_dfs_pipe.items():
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    all_dfs = all_dfs.drop_duplicates('Opportunity ID', keep = 'last')
    pipe_sf = all_dfs.reset_index(drop=True)

    empid_gc_map = map_empid_growth_cell()
    pipe_sf['Growth Cell'] = pipe_sf['Opportunity Lead: Employee Number'].apply(lambda x: empid_gc_map[x] if x in empid_gc_map.keys() else 'Cannot Find')
    pipe_sf['Created_date_year'] = pipe_sf['Created Date'].apply(lambda x: x.year)
    pipe_sf['Created_date_month'] = pipe_sf['Created Date'].apply(lambda x: x.month)

    pipe_sf_agg = pipe_sf[pipe_sf['Created Date']> datetime.datetime(2021, 7, 1)].groupby(['Growth Cell', 'Created_date_year', 'Created_date_month']).agg({'Amount (converted)': 'sum', 'Opportunity ID':'count', 'Ultimate Parent Account Name':pd.Series.nunique}) .reset_index()
    for gc in pipe_sf_agg['Growth Cell'].unique():
        k = pipe_sf_agg[pipe_sf_agg['Growth Cell'] == gc]
        for i,row in pipe_sf_agg[['Created_date_year','Created_date_month']].drop_duplicates().iterrows():
            if len(k[(k['Created_date_year'] == row['Created_date_year']) & (k['Created_date_month'] == row['Created_date_month'])])<1:
                pipe_sf_agg = pipe_sf_agg.append({'Growth Cell':gc, 'Created_date_year': row['Created_date_year'], 'Created_date_month': row['Created_date_month'], 'Amount (converted)':0, 'Opportunity ID': 0, 'Ultimate Parent Account Name': 0}, ignore_index = True)
    pipe_sf_agg = pipe_sf_agg.sort_values(['Growth Cell', 'Created_date_year', 'Created_date_month'])
    pipe_sf_agg = pipe_sf_agg.rename(columns = {'Created_date_year':'Year', 'Created_date_month':'Month'})

    return pipe_sf_agg

def utilization_trends(path = os.path.join(sai_dir,'07.01.22.xlsx - Export.csv')):

    utilizn = pd.read_csv(path)
    utilizn = utilizn[~utilizn['Workday ID'].isna()]
    empid_gc_map = map_empid_growth_cell()
    empid_gc_map = {str(k):v for k,v in empid_gc_map.items()}
    utilizn['Growth Cell'] = utilizn['Workday ID'].apply(lambda x: empid_gc_map[str(int(x))] if str(int(x)) in empid_gc_map.keys() else "Cannot Find")
    utilizn_growth_cell = utilizn.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': lambda x: avg_util(x),
                                                                       '21-Aug': lambda x: avg_util(x),
                                                                       '21-Sep': lambda x: avg_util(x),
                                                                       '21-Oct': lambda x: avg_util(x),
                                                                       '21-Nov': lambda x: avg_util(x),
                                                                       '21-Dec': lambda x: avg_util(x),
                                                                       '22-Jan': lambda x: avg_util(x),
                                                                       '22-Feb': lambda x: avg_util(x),
                                                                       '22-Mar': lambda x: avg_util(x),
                                                                       '22-Apr': lambda x: avg_util(x),
                                                                       '22-May': lambda x: avg_util(x),
                                                                       '22-Jun': lambda x: avg_util(x)
                                                                       }).reset_index()
    utilizn_growth_cell_count = utilizn.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': lambda x: count_resources(x),
                                                                       '21-Aug': lambda x: count_resources(x),
                                                                       '21-Sep': lambda x: count_resources(x),
                                                                       '21-Oct': lambda x: count_resources(x),
                                                                       '21-Nov': lambda x: count_resources(x),
                                                                       '21-Dec': lambda x: count_resources(x),
                                                                       '22-Jan': lambda x: count_resources(x),
                                                                       '22-Feb': lambda x: count_resources(x),
                                                                       '22-Mar': lambda x: count_resources(x),
                                                                       '22-Apr': lambda x: count_resources(x),
                                                                       '22-May': lambda x: count_resources(x),
                                                                       '22-Jun': lambda x: count_resources(x)
                                                                       }).reset_index()

    utilizn_growth_cell_util_diff = calc_util_diff(utilizn)

    utilizn_growth_cell_mean_over = utilizn_growth_cell_util_diff.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': avg_above,
                                                                       '21-Aug': avg_above,
                                                                       '21-Sep': avg_above,
                                                                       '21-Oct': avg_above,
                                                                       '21-Nov': avg_above,
                                                                       '21-Dec': avg_above,
                                                                       '22-Jan': avg_above,
                                                                       '22-Feb': avg_above,
                                                                       '22-Mar': avg_above,
                                                                       '22-Apr': avg_above,
                                                                       '22-May': avg_above,
                                                                       '22-Jun': avg_above
                                                                       }).reset_index()


    utilizn_growth_cell_mean_below = utilizn_growth_cell_util_diff.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': avg_below,
                                                                       '21-Aug': avg_below,
                                                                       '21-Sep': avg_below,
                                                                       '21-Oct': avg_below,
                                                                       '21-Nov': avg_below,
                                                                       '21-Dec': avg_below,
                                                                       '22-Jan': avg_below,
                                                                       '22-Feb': avg_below,
                                                                       '22-Mar': avg_below,
                                                                       '22-Apr': avg_below,
                                                                       '22-May': avg_below,
                                                                       '22-Jun': avg_below
                                                                       }).reset_index()


    utilizn_growth_cell_count_above = utilizn_growth_cell_util_diff.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': count_above,
                                                                       '21-Aug': count_above,
                                                                       '21-Sep': count_above,
                                                                       '21-Oct': count_above,
                                                                       '21-Nov': count_above,
                                                                       '21-Dec': count_above,
                                                                       '22-Jan': count_above,
                                                                       '22-Feb': count_above,
                                                                       '22-Mar': count_above,
                                                                       '22-Apr': count_above,
                                                                       '22-May': count_above,
                                                                       '22-Jun': count_above
                                                                       }).reset_index()


    utilizn_growth_cell_count_below = utilizn_growth_cell_util_diff.groupby(['Growth Cell', 'Level'])[['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
       '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']].agg({'21-Jul': count_below,
                                                                       '21-Aug': count_below,
                                                                       '21-Sep': count_below,
                                                                       '21-Oct': count_below,
                                                                       '21-Nov': count_below,
                                                                       '21-Dec': count_below,
                                                                       '22-Jan': count_below,
                                                                       '22-Feb': count_below,
                                                                       '22-Mar': count_below,
                                                                       '22-Apr': count_below,
                                                                       '22-May': count_below,
                                                                       '22-Jun': count_below
                                                                       }).reset_index()

    utilizn_growth_cell = transpose_df(utilizn_growth_cell, '_Util')
    utilizn_growth_cell_count = transpose_df(utilizn_growth_cell_count, '_Count')
    utilizn_growth_cell_mean_over = transpose_df(utilizn_growth_cell_mean_over, '_Mean_Above')
    utilizn_growth_cell_mean_below = transpose_df(utilizn_growth_cell_mean_below, '_Mean_Below')
    utilizn_growth_cell_count_above = transpose_df(utilizn_growth_cell_count_above, '_Count_Above')
    utilizn_growth_cell_count_below = transpose_df(utilizn_growth_cell_count_below, '_Count_Below')

    return utilizn_growth_cell.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True), utilizn_growth_cell_count.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True), utilizn_growth_cell_mean_over.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True), utilizn_growth_cell_mean_below.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True), utilizn_growth_cell_count_above.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True), utilizn_growth_cell_count_below.sort_values(['Growth Cell', 'Year', 'Month']).reset_index(drop = True)



def avg_util(x):
    k = [int(''.join(t for t in str(y) if t.isdigit())) for y in x if len(''.join(t for t in str(y) if t.isdigit()))>0]
    return sum(k)/len(k) if len(k)>0 else 0

def count_resources(x):
    k = [int(''.join(t for t in str(y) if t.isdigit())) for y in x if len(''.join(t for t in str(y) if t.isdigit()))>0]
    return len(k)

def diff(a, b):
    try:
        return int(a.replace('%','')) - int(b.replace('%',''))
    except:
        return np.nan

def calc_util_diff(df):
    months = ['21-Jul', '21-Aug', '21-Sep', '21-Oct', '21-Nov', '21-Dec', '22-Jan',
        '22-Feb', '22-Mar', '22-Apr', '22-May', '22-Jun']

    for month in months:
        df[month] = df[['Util Target', month]].apply(lambda x: diff(x['Util Target'], x[month]), axis = 1)
    
    return df

def avg_above(x):
    return x[x>0].mean()

def avg_below(x):
    return x[x<0].mean()

def count_above(x):
    return x[x>0].count()

def count_below(x):
    return x[x<0].count()

def transpose_df(df, suffix):
    df['Level'] = df['Level'] + suffix
    pivot_df = df.pivot(index='Growth Cell', columns='Level', values=['21-Jul', '21-Aug', '21-Sep', '21-Oct',
       '21-Nov', '21-Dec', '22-Jan', '22-Feb', '22-Mar', '22-Apr', '22-May',
       '22-Jun'])
    pivot_df = pivot_df.stack(0).reset_index()
    pivot_df[['Year','Month']]= pivot_df['level_1'].str.split(pat='-',expand=True)
    pivot_df['Year'] = pivot_df['Year'].map(year_mappings)
    pivot_df['Month'] = pivot_df['Month'].map(month_mappings)
    pivot_df['Year'] = pivot_df['Year'].astype('Int64')
    pivot_df['Month'] = pivot_df['Month'].astype('Int64')
    # pivot_df = pivot_df.drop('Year',1)
    # pivot_df = pivot_df.drop('Month',1)
    pivot_df = pivot_df.drop('level_1',1)
    
    return pivot_df


year_mappings = {'21':2021, '22':2022}


month_mappings = {'Jul': 7,
                 'Aug': 8, 
                 'Sep': 9, 
                 'Oct': 10, 
                 'Nov': 11, 
                 'Dec': 12, 
                 'Jan': 1, 
                 'Feb': 2,
                 'Mar': 3,
                 'Apr': 4,
                 'May': 5,
                 'Jun': 6}

def create_identifier(df):
    df['Identifier'] = df['Growth Cell'] + '_' + df['Month'].astype(float).astype(str) + '_' + df['Year'].astype(float).astype(str)
    
    return df

def create_all_data(all_data):
    
    for df in all_data:
        df = create_identifier(df)
    
    all_data_df = all_data[0]

    for df in all_data[1:]:
        all_data_df = all_data_df.merge(df, on='Identifier', how = 'outer')

    all_data_df = all_data_df.drop(columns = ['Year_x', 'Month_x', 'Year_y', 'Month_y',
     'Specialist_Util', 'Managing Director_Count_Above', 'Partner_Count_Above', 'Managing Director_Count_Below', 'Partner_Count_Below',
      'Specialist_Count_Below', 'Managing Director_Mean_Above', 'Partner_Mean_Above','Specialist_Mean_Above',
      'Managing Director_Mean_Below', 'Partner_Mean_Below','Specialist_Mean_Below','Managing Director_Count',
       'Partner_Count', 'Specialist_Count','Specialist_Count_Above','Growth Cell_y', 'Growth Cell_x', 'Managing Director_Util'])
    
    all_data_df[['Growth Cell','Time']]= all_data_df['Identifier'].str.split('_',1,expand=True)
    #all_data_df = all_data_df[all_data_df.apply(lambda x: sum(x.isna())/len(x) <.06, axis = 1)]

    #imputing
    '''all_data_df['Associate_Mean_Above'] = all_data_df['Associate_Mean_Above'].fillna(all_data_df['Associate_Mean_Above'].mean())
    all_data_df['Director_Mean_Above'] = all_data_df['Director_Mean_Above'].fillna(all_data_df['Director_Mean_Above'].mean())
    all_data_df['Manager_Mean_Above'] = all_data_df['Manager_Mean_Above'].fillna(all_data_df['Manager_Mean_Above'].mean())
    all_data_df['Senior Associate_Mean_Above'] = all_data_df['Senior Associate_Mean_Above'].fillna(all_data_df['Senior Associate_Mean_Above'].mean())
    all_data_df['Senior Manager_Mean_Above'] = all_data_df['Senior Manager_Mean_Above'].fillna(all_data_df['Senior Manager_Mean_Above'].mean())
    all_data_df['Associate_Mean_Below'] = all_data_df['Associate_Mean_Below'].fillna(all_data_df['Associate_Mean_Below'].mean())
    all_data_df['Director_Mean_Below'] = all_data_df['Director_Mean_Below'].fillna(all_data_df['Director_Mean_Below'].mean())
    all_data_df['Manager_Mean_Below'] = all_data_df['Manager_Mean_Below'].fillna(all_data_df['Manager_Mean_Below'].mean())
    all_data_df['Senior Manager_Mean_Below'] = all_data_df['Senior Manager_Mean_Below'].fillna(all_data_df['Senior Manager_Mean_Below'].mean())'''


    return all_data_df


def revenue(path_to_folder = os.path.join(sai_dir,'Monthly Performance Data'), path_to_folder_2 = os.path.join(sai_dir,'P_MD_Constants')):

    #name_gc_map = map_name_growth_cell_()
    all_dfs_rev = {}
    for file in os.listdir(path=path_to_folder):
        all_dfs_rev[file] = pd.read_excel(os.path.join(path_to_folder, file))
    
    all_dfs = []
    for file, df in all_dfs_rev.items():
        file_gc = file.split(".")[0] + ".csv"
        if file_gc in os.listdir(path = path_to_folder_2):
            name_gc_map = map_name_growth_cell_(os.path.join(path_to_folder_2, 'constants.csv'))
        else:
            name_gc_map = map_name_growth_cell_(os.path.join(path_to_folder_2, 'constants.csv'))
        df['Growth Cell'] = df['Engagement Partner'].apply(lambda x: name_gc_map[x] if x in name_gc_map.keys() else 'Cannot Find')
        k = df.groupby('Growth Cell').agg({'Monthly Revenue': ['sum', 'mean'], 'Monthly EM%':'mean'}).reset_index()
        k['period'] = file
        all_dfs.append(k)
    
    all_dfs = pd.concat(all_dfs)
    all_dfs['period'] = all_dfs['period'].apply(lambda x: x.split('.')[0])

    all_dfs['year'] = pd.to_datetime(all_dfs['period']).apply(lambda x: x.year)
    all_dfs['month'] = pd.to_datetime(all_dfs['period']).apply(lambda x: x.month)

    all_dfs = pd.concat([all_dfs['Growth Cell'], all_dfs['Monthly Revenue']['sum'], all_dfs['Monthly EM%']['mean'], all_dfs['year'], all_dfs['month']], axis = 1)
    all_dfs.columns = ['Growth Cell', 'Revenue Sum', 'EM% mean', 'Year', 'Month']
    all_dfs = all_dfs.sort_values(['Growth Cell', 'Year', 'Month'])

    return all_dfs

def moving_avg_features(list_of_tables, roll_window = 2):
    new_dfs = []
    for df in list_of_tables:
        df = df[~df[['Year', 'Month']].apply(lambda x: any(x.isna()), axis = 1)]
        df = df.sort_values(['Growth Cell', 'Year', 'Month'])
        agg_columns = list(set(df.columns.tolist()) - set(['Growth Cell', 'Year', 'Month', 'Identifier']))
        '''df[agg_columns] = (df[agg_columns] - df[agg_columns].shift())*100/df[agg_columns]
        mask = df['Growth Cell'] != df['Growth Cell'].shift()
        for col in agg_columns:
            df[col].loc[mask] = np.nan
        df = df[~df[agg_columns].apply(lambda x: all(x.isna()), axis = 1)]'''
        k = df.groupby('Growth Cell')[agg_columns].rolling(window=roll_window).mean().reset_index()
        k['Year'] = df['Year'].values
        k['Month'] = df['Month'].values
        '''f = []
        for gc in k['Growth Cell'].unique():
            b = k[k['Growth Cell'] == gc]
            b[agg_columns] = b[agg_columns].shift(+1)
            f.append(b)
        l = pd.concat(f)'''
        l = k
        l = l[~l[agg_columns].apply(lambda x: all(x.isna()), axis = 1)]
        l = l.drop('level_1', axis = 1)
        new_dfs.append(l)
    return new_dfs

def moving_avg_features_pc(list_of_tables, roll_window = 2):
    new_dfs = []
    for df in list_of_tables:
        df = df[~df[['Year', 'Month']].apply(lambda x: any(x.isna()), axis = 1)]
        df = df.sort_values(['Growth Cell', 'Year', 'Month'])
        agg_columns = list(set(df.columns.tolist()) - set(['Growth Cell', 'Year', 'Month', 'Identifier']))
        df[agg_columns] = (df[agg_columns] - df[agg_columns].shift())*100/df[agg_columns]
        mask = df['Growth Cell'] != df['Growth Cell'].shift()
        for col in agg_columns:
            df[col].loc[mask] = np.nan
        df = df[~df[agg_columns].apply(lambda x: all(x.isna()), axis = 1)]
        k = df.groupby('Growth Cell')[agg_columns].rolling(window=roll_window).mean().reset_index()
        k['Year'] = df['Year'].values
        k['Month'] = df['Month'].values
        f = []
        for gc in k['Growth Cell'].unique():
            b = k[k['Growth Cell'] == gc]
            b[agg_columns] = b[agg_columns].shift(+1)
            f.append(b)
        l = pd.concat(f)
        l = l[~l[agg_columns].apply(lambda x: all(x.isna()), axis = 1)]
        l = l.drop('level_1', axis = 1)
        l.columns = [x+'_pc' if x in agg_columns else x for x in l.columns ]
        new_dfs.append(l)
    return new_dfs

def rev_lagged_features(df_rev):
    agg_columns = list(set(df_rev.columns.tolist()) - set(['Growth Cell', 'Year', 'Month', 'Identifier']))

    '''df_rev[agg_columns] = (df_rev[agg_columns] - df_rev[agg_columns].shift())*100/df_rev[agg_columns]
    mask = df_rev['Growth Cell'] != df_rev['Growth Cell'].shift()
    for col in agg_columns:
        df_rev[col].loc[mask] = np.nan
    df_rev = df_rev[~df_rev[agg_columns].apply(lambda x: all(x.isna()), axis = 1)]'''

    k1 = df_rev.groupby('Growth Cell')[agg_columns].shift(+1).reset_index(drop = True)
    k1['Growth Cell'] = df_rev['Growth Cell'].tolist()

    k1.columns = ['EM% mean - 1', 'Revenue Sum - 1', 'Growth Cell']

    k1['Year'] = df_rev['Year'].values
    k1['Month'] = df_rev['Month'].values

    k1 = k1[~k1[['EM% mean - 1', 'Revenue Sum - 1']].apply(lambda x: all(x.isna()), axis = 1)]
    return k1


def rev_to_pred(df_rev, t_step = 1):
    agg_columns = list(set(df_rev.columns.tolist()) - set(['Growth Cell', 'Year', 'Month', 'Identifier']))

    k1 = df_rev.groupby('Growth Cell')[agg_columns].shift(-t_step).reset_index(drop = True)
    k1['Growth Cell'] = df_rev['Growth Cell'].tolist()

    k1 = k1.rename(columns = {'Revenue Sum':'Revenue Sum + 1', 'EM% mean': 'EM% mean + 1'})

    k1['Year'] = df_rev['Year'].values
    k1['Month'] = df_rev['Month'].values

    k1 = k1[~k1[['Revenue Sum + 1', 'EM% mean + 1']].apply(lambda x: all(x.isna()), axis = 1)]
    return k1