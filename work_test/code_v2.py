import pandas as pd

df_full = pd.read_csv("input.csv")

for col in df_full:
    if isinstance(df_full[col][0],str):
        df_full[col] = df_full[col].apply(lambda x: x.strip())
        
df = df_full.copy()

df['imageID_13_char'] = df['Image_ID'].apply(lambda x: x[:13])

df['total'] = df.apply(lambda x: x['include']+x['date']+x['State']+x['complete']+x['version'],axis=1)

d = df.groupby(['total'])[['total','imageID_13_char']].indices

dup_indx = []
for indx in d.values():
    if len(indx)>1:
        dup_indx = dup_indx + list(indx)

df = df[list(df)[:-1]].iloc[dup_indx]
df.reset_index(inplace=True,drop=True)
df_file1 = df[['include','date','State','complete','version','imageID_13_char']]
df_file1 = df_file1.drop_duplicates(keep=False)

df_file1 = df.iloc[df_file1.index]
df_file1 = df_file1.drop('imageID_13_char',axis=True)

dup_indx = list(set(df.index) - set(df_file1.index))

df_file2 = df.iloc[dup_indx]
df_file2 = df_file2.drop('imageID_13_char',axis=True)

df_file1.reset_index(inplace=True,drop=True)
df_file2.reset_index(inplace=True,drop=True)

# New code Here
df = df_file1.copy()

df['conf_level'] = df['conf_level'].replace(['high','moderate','low'],[3,2,1])

df['total'] = df.apply(lambda x: x['include']+str(x['date'])+x['State']+x['complete']+x['version'],axis=1)

d = df.groupby(['total'])[['total']].indices

indx_drop = []

for indx in d.values():
    if len(indx)>1:
        x = set(df['Quantity'].iloc[indx])
        if len(x)==1:
            indx = df['conf_level'].iloc[indx].idxmax()
        else:
            indx = df['Quantity'].iloc[indx].idxmax()
        indx_drop.append(indx)
    
df_result1 = pd.concat([df_file2,df_file1.iloc[indx_drop]]).reset_index(drop=True)
df_result2 = df_file1.drop(indx_drop)

df_result1.to_csv('Result1.csv',index=False) 
df_result2.to_csv('Result2.csv',index=False)