#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv('E:\OneDrive - NITT\Custom_Download\monthly_volume_0611.csv')
print(len(df.index))

#cleaning
df.drop(df[df['distributor_name'] == "TestWD"].index, inplace = True)
df.drop(df[df['customer_name'] == "Bankers Hill Food"].index, inplace = True)
df.drop(df[df['customer_name'] == "Shahi Feed"].index, inplace = True)
len(df.index)

df.customer_name=df['customer_name'].str.strip()
df.distributor_name=df['distributor_name'].str.strip()
df.loc[df.ticket_id=="ID-1648","scheduled_date"]="2020-09-18"
df['datef']=df['scheduled_date'].apply(str)
df['datef']=df['scheduled_date'].str[:7]
df['datef']=df['datef'].apply(str)

#Rename
df.loc[(df.distributor_name == "Ghaziabad Warehouse") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL WH Ghaziabad"
df.loc[(df.distributor_name == "Hamidpur Warehouse") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL WH Hamidpur"
df.loc[(df.distributor_name == "Hassangarh Warehouse") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL WH Hassangarh"
df.loc[(df.distributor_name == "Hassangarh Warehouse") & (df.customer_name == "ITC Ltd. Haryana") , 'customer_name'] = "ITC DEL WH Hassangarh"
df.loc[(df.distributor_name == "MJ Warehousing Pvt Ltd") & (df.customer_name == "ITC Ltd. Haryana") , 'customer_name'] = "ITC DEL WH Jhundpur"
df.loc[(df.distributor_name == "MJ Warehousing Pvt Ltd") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL WH Jhundpur"
df.loc[(df.customer_name == "ITC Ltd. FBD") , 'customer_name'] = "ITC DEL WH Jhundpur"
df.loc[(df.distributor_name == "Sourcewell Agro Foods Pvt Ltd") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC FBD Sourcewell"
df.loc[(df.distributor_name == "01 Sourcewell Agro Foods Pvt Ltd") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC FBD Sourcewell"
df.loc[(df.distributor_name == "Ambarnath Warehouse") & (df.customer_name == "ITC Ltd. Mumbai") , 'customer_name'] = "ITC MUM WH Ambernath"
df.loc[(df.distributor_name == "ITC Ray Road Warehouse") & (df.customer_name == "ITC Ltd. Mumbai") , 'customer_name'] = "ITC MUM WH Ray Road"
df.loc[(df.distributor_name == "Narela Warehouse") & (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL WH Narela"
df.loc[(df.distributor_name == "Lucknow Warehouse") & (df.customer_name == "ITC Ltd. Lucknow") , 'customer_name'] = "ITC UP WH Lucknow"
df.loc[(df.distributor_name == "Varanasi Warehouse") & (df.customer_name == "ITC Ltd. Lucknow") , 'customer_name'] = "ITC UP WH Varanasi"
df.loc[(df.distributor_name == "Kanpur Warehouse") & (df.customer_name == "ITC Ltd. Lucknow") , 'customer_name'] = "ITC UP WH Kanpur"
df.loc[(df.distributor_name == "Bengaluru Warehouse") & (df.customer_name == "Veeba Food Services Pvt Ltd. Bangalore") , 'customer_name'] = "Veeba Food Services Bangalore WH"
df.loc[(df.distributor_name == "Keshwana Warehouse") & (df.customer_name == "Veeba Food Services Pvt Ltd. Rajasthan") , 'customer_name'] = "Veeba Food Services DEL WH Keshwana"
df.loc[(df.distributor_name == "Neemrana Warehouse") & (df.customer_name == "Veeba Food Services Pvt Ltd. Rajasthan") , 'customer_name'] = "Veeba Food Services DEL WH Neemarana"
df.loc[(df.distributor_name == "Tikri Warehouse") & (df.customer_name == "Veeba Food Services Pvt Ltd") , 'customer_name'] = "Veeba Food Services DEL WH Tikri"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Mumbai") , 'customer_name'] = "General Mills India Maharashtra"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Bangalore") , 'customer_name'] = "General Mills India Karnataka"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Tamilnadu") , 'customer_name'] = "General Mills India Tamil Nadu"
df.loc[(df.customer_name == "General Mills India Pvt Ltd.") , 'customer_name'] = "General Mills India Delhi"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Punjab") , 'customer_name'] = "General Mills India Punjab"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Lucknow") , 'customer_name'] = "General Mills India Uttar Pradesh"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. West Bengal") , 'customer_name'] = "General Mills India West Bengal"
df.loc[(df.customer_name == "General Mills India Pvt Ltd. Telangana") , 'customer_name'] = "General Mills India Telangana"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Mumbai") & (df.distributor_name == "Unibic Foods India Pvt. Ltd") , 'customer_name'] = "Unibic Foods India Mumbai"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Mumbai") & (df.city == "Mumbai") , 'customer_name'] = "Unibic Foods India Mumbai"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Mumbai") , 'customer_name'] = "Unibic Foods India Rest of Maharashtra"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd.") & (df.distributor_name == "Unibic Foods India Pvt. Ltd") , 'customer_name'] = "Unibic Foods India Delhi"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Madhya Pradesh") & (df.distributor_name == "Unibic Foods India Pvt. Ltd") , 'customer_name'] = "Unibic Foods India Madhya Pradesh"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Gujrat") & (df.distributor_name == "Unibic Foods India Pvt. Ltd") , 'customer_name'] = "Unibic Foods India Gujarat"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Punjab") & (df.distributor_name == "Unibic Foods India Pvt. Ltd"), 'customer_name'] = "Unibic Foods India Punjab"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Bangalore") & (df.distributor_name == "Unibic Foods India Pvt. Ltd"), 'customer_name'] = "Unibic Foods India Karnataka"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Andhra Pradesh") & (df.distributor_name == "Unibic Foods India Pvt. Ltd"), 'customer_name'] = "Unibic Foods India Andhara"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Telangana") & (df.distributor_name == "Unibic Hyderabad CFA"), 'customer_name'] = "Unibic Foods India Telangana"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd.") , 'customer_name'] = "Unibic Foods India Delhi"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Madhya Pradesh") , 'customer_name'] = "Unibic Foods India Madhya Pradesh"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Gujrat")  , 'customer_name'] = "Unibic Foods India Gujarat"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Punjab") , 'customer_name'] = "Unibic Foods India Punjab"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Bangalore") , 'customer_name'] = "Unibic Foods India Karnataka "
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Andhra Pradesh") , 'customer_name'] = "Unibic Foods India Andhara"
df.loc[(df.customer_name == "Unibic Foods India Pvt. Ltd. Goa") , 'customer_name'] = "Unibic Foods India Goa"
df.loc[ (df.customer_name == "ITC Ltd.") , 'customer_name'] = "ITC DEL Distributors"
df.loc[ (df.customer_name == "ITC Ltd. Mumbai") , 'customer_name'] = "ITC Mumbai Distributors"
df.loc[ (df.customer_name == "ITC Ltd. Lucknow") , 'customer_name'] = "ITC Uttar Pradesh Distributors"
df.loc[ (df.customer_name == "ITC Ltd. Haryana") , 'customer_name'] = "ITC Panchkula"
df.loc[ (df.customer_name == "ITC Ltd. Panchkula") , 'customer_name'] = "ITC Panchkula"
df.loc[(df.customer_name == "Veeba Food Services Pvt Ltd") , 'customer_name'] = "Veeba Food Services DEL Distributors"
df.loc[(df.customer_name == "Veeba Food Services Pvt Ltd. Mumbai") , 'customer_name'] = "Veeba Food Services Maharashtra Distributors"
df.loc[(df.customer_name == "Veeba Food Services Pvt Ltd. Lucknow") , 'customer_name'] = "Veeba Food Services Uttar Pradesh Distributors"
df.loc[(df.customer_name == "Veeba Food Services Pvt Ltd. Bangalore") , 'customer_name'] = "Veeba Food Services Karnataka Distributors"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Jharkhand") , 'customer_name'] = "Tata Consumer Products Jharkhand"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Bihar") , 'customer_name'] = "Tata Consumer Products Bihar"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Uttar Pradesh") , 'customer_name'] = "Tata Consumer Products Uttar Pradesh"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Karnataka") , 'customer_name'] = "Tata Consumer Products Karnataka"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Bangalore") , 'customer_name'] = "Tata Consumer Products Karnataka"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Gujarat") , 'customer_name'] = "Tata Consumer Products Gujarat"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Telangana") , 'customer_name'] = "Tata Consumer Products Telangana"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. West Bengal") , 'customer_name'] = "Tata Consumer Products West Bengal"
df.loc[(df.customer_name == "Tata Consumer Products Ltd. Madhya Pradesh") , 'customer_name'] = "Tata Consumer Products Madhya Pradesh"
df.loc[(df.customer_name == "Vishv Foods and Beverages LLP") , 'customer_name'] = "Vishv Foods and Beverages (Snackry)"

#filter supplier
# new_df=df[df['customer_name'].str.contains('Tata Consumer Products Mad')]
# new_df['scheduled_date'] = new_df['scheduled_date'].apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))
# new_df['scheduled_date']=pd.to_datetime(new_df['scheduled_date'], format='%d/%m/%Y')
# new_df['scheduled_date']=(pd.to_datetime(new_df['scheduled_date']) - pd.datetime(1899, 12, 30)) / pd.to_timedelta(1, unit='D')


# In[2]:


import calendar
df['scheduled_date'] =  pd.to_datetime(df['scheduled_date'])
df['Month'] = df['scheduled_date'].dt.month
df['Year'] = df['scheduled_date'].dt.year
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])


# In[3]:


def get_data(companies,year,month,th):
    try:
        df_custom = df[(df['customer_name'].isin(companies)) & (df['Year']==year) & (df['Month']==month)]
        total_wt = sum(df_custom['item_weight'])
        x = round(df_custom[['item_name','item_weight']].groupby('item_name').sum()*100/total_wt,3)
        x.reset_index(inplace=True)
        y = df_custom[['distributor_name','item_weight']].groupby('distributor_name').sum()
        y.reset_index(inplace=True)
        y = y.sort_values('item_weight',ascending=False)[:5]
        y.columns = ['distributor_name','Item_Weight']
        fw = total_wt*(100-th)/100
        df1 = pd.DataFrame({'Total_Item_weights':[total_wt],'Packaging_%val_item_wt':[th],
                            'Food_Weight':[fw],'Non_Food_Weight':[total_wt*th/100],'No of pickups':[df_custom['ticket_id'].nunique()],
                           'Unique location': [df_custom['distributors_code'].nunique()] })
        df2 = pd.concat([df1,x,y], ignore_index=True)
        
        writer = pd.ExcelWriter(f'{companies[0].split()[0]+month+str(year)}.xlsx', engine='xlsxwriter')

        # Write each dataframe to a different worksheet. you could write different string like above if you want
        df2.to_excel(writer, sheet_name='processed',startcol=1,index=False)
        df_custom.to_excel(writer, sheet_name='raw',startcol=2,index=False)
        writer.save()
    except:
        return 'Wrong parameters'


# In[4]:


df['customer_name'].value_counts()


# In[5]:


get_data(['ITC DEL Distributors'],2022,'May',10)


# In[6]:


get_data(['ITC DEL Distributors','ITC Uttar Pradesh Distributors'],2022,'May',10)

