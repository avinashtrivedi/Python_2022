import pandas as pd
df_file1 = pd.read_excel('input.xlsx',sheet_name='file1')
df_file2 = pd.read_excel('input.xlsx',sheet_name='file2')

df = df_file1.copy()

for col in df:
    if isinstance(df[col][0],str):
        df[col] = df[col].apply(lambda x: x.strip())

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
writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')
df_result1.to_excel(writer, sheet_name='result1',index=False)
df_result2.to_excel(writer, sheet_name='result2',index=False)
writer.save()