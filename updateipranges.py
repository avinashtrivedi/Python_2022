from genericpath import exists
import requests
import os
import json
import csv
import pandas as pd
import urllib.request
from jblib import hilight as hl

debug = True
environment = 'prod'

if environment == 'prod':
    baseURL = 'https://onecloud.comcast.net'
else:
    baseURL = f'https://{environment}.onecloud.comcast.net'

# Load OneCloud JWT
token = os.environ.get("JWT_TOKEN")
# Set up Authorization header with OneCloud JWT
headers = {"Authorization": "Bearer " + token}

df = pd.read_csv('input.csv')
exists_error = "another tenant with the name"
data_all = []
for index, row in df.iterrows():
    tenantId = row["tenantId"]

    # use swagger API to check the status of the tenant
    resp = requests.get(f"{baseURL}/api/tsf/v2/tenants/{row['tenantId']}",headers=headers).json()

    if resp.get("status") == "active":
        print(f'\t{hl("Tenant:").blue(True)} {row["tenantId"]} - {hl("ACTIVE").green(True)}')
        response = requests.get(f"{baseURL}/api/tsf/v2//tenants/{row['tenantId']}/networks/external",headers=headers).json()

        info = [(response[0]['tsfTenantID'], d['networkID'],d['cidrBlock']) for d in response[0]['cidrs']]
        df = pd.DataFrame([],columns=['tsfTenantID','networkID','cidrBlock'])
        for detail in info:
            df.loc[len(df)] = detail
        data_all.append(df)
        
df = pd.concat(data_all)
df.to_csv('onecloud_output.csv',index=False)

with urllib.request.urlopen("https://ip-ranges.amazonaws.com/ip-ranges.json") as url:
    aws_ip_ranges_resp = json.loads(url.read().decode())

data_ipv4 = [(d['ip_prefix'], d['region'],d['service']) for d in aws_ip_ranges_resp['prefixes']]
data_ipv6 = [(d['ipv6_prefix'], d['region'],d['service']) for d in aws_ip_ranges_resp['ipv6_prefixes']]
df_ipv4 = pd.DataFrame([],columns=['iprange','region','service'])
df_ipv6 = pd.DataFrame([],columns=['iprange','region','service'])

for detail in data_ipv4:
    df_ipv4.loc[len(df_ipv4)] = detail

for detail in data_ipv6:
    ipv6.loc[len(ipv6)] = detail
    
df_ipv4.to_csv('aws_ipv4_output.csv',index=False)
df_ipv6.to_csv('aws_ipv6_output.csv',index=False)

#compare this two csv 