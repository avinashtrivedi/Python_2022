import pandas as pd
import requests

def get_url(record):
    street=''
    if record['Building_Number']:
        street+=record['Building_Number'][:-2]+' '
    if record['Building_Name']:
        street+=record['Building_Name']+' '
    if record['Street_Name']:
        street+=record['Street_Name']+' '
    street=street.strip()
    city=record['City']
    state=record['State']
    country=record['Country']
    postalcode=record['Zip_Code']
    result=url
    if street:
        result=result+'street='+street+'&'
    if city:
        result=result+'city='+city+'&'
    if state:
        result=result+'state='+state+'&'
    if country:
        result=result+'country='+country+'&'
    if postalcode:
        result=result+'postalcode='+postalcode+'&'
    query=requests.get(result[:-1])
    print(query.text!='[]')
    return query.text!='[]'

# Loading data
train=pd.read_csv("us-train-dataset.csv")
test=pd.read_csv("us-test-dataset.csv")

train.fillna('', inplace=True)
test.fillna('', inplace=True)

train['Building_Number']=train['Building_Number'].astype('str')
test['Building_Number']=test['Building_Number'].astype('str')
train['Building_Name']=train['Building_Name'].astype('str')
test['Building_Name']=test['Building_Name'].astype('str')

url="https://nominatim.openstreetmap.org/search.php?format=jsonv2&"

train['label']=train.apply(get_url,axis=1).astype('int')
test['label']=test.apply(get_url,axis=1).astype('int')

# Writing data with labels to csv file
train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)