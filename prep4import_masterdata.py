#!/usr/bin/env python3
"""prep4import_masterdata.py

Processes Level-1 dataset from Parquet files to Level-2 ready for import to the
Neo4j graph database. This dataset is called MasterData because it contains only
Company, Facility, Location, LOCATED_IN, and BELONGS_TO entities.

"""
# %% DEPENDENCIES

import os
import sys
import glob
from collections import Counter
from functools import reduce
from pathlib import Path
sys.path.append('/Users/dvagal/PycharmProjects/snd_pipeline/suppliernetworkdiscovery-dev-rj/')
sys.path.append('/Users/dvagal/PycharmProjects/snd_pipeline/suppliernetworkdiscovery-dev-rj/snd/')
import pandas as pd
import numpy as np
import csv
from snd.data.resolve.canonical import resolve_canonicals
import util

# %% DATA FILE DIRECTORY
# @TODO: move this to Yaml configuration file

# DATA_PATH = Path("snd/data")
# INTERMEDIATE_DATASETS = DATA_PATH / "intermediate_data"
# LEVEL1_PATH = INTERMEDIATE_DATASETS / "Level1/MasterData"
# LEVEL2_PATH = INTERMEDIATE_DATASETS / "Level2/MasterData"

CURDIR = os.curdir
print(CURDIR)
DATA_PATH = os.path.join(CURDIR,os.path.join('snd','data'))
INTERMEDIATE_DATASETS = os.path.join(DATA_PATH,'intermediate_data')
LEVEL1_PATH = os.path.join(os.path.join(INTERMEDIATE_DATASETS,'Level1'),'MasterData')
LEVEL2_PATH = os.path.join(os.path.join(INTERMEDIATE_DATASETS,'Level2'),'MasterData')


os.makedirs(LEVEL2_PATH,exist_ok=True)
# %% DATASETS
# Read data from files
data = []
print(os.path.join(LEVEL1_PATH, "part-00000-58855cb4-2cb6-4f88-92b3-bc66a2eae7ed-c000.snappy.parquet"))

filename = '/Users/dvagal/PycharmProjects/snd_pipeline/suppliernetworkdiscovery-dev-rj/snd/data/intermediate_data/Level1/MasterData/part-00000-58855cb4-2cb6-4f88-92b3-bc66a2eae7ed-c000.snappy.parquet'

data.append(pd.read_parquet(filename))

print('--------------')
print(len(data))
data = pd.concat(data)
print("shape =", data.shape)
print("schema:")
print(data.dtypes)

# %% DATA SCHEMA
# Field names of interest

COMPANY_Z_CLUSTER = "z_cluster"
COMPANY_Z_MAXSCORE = "z_maxscore"
COMPANY_Z_MINSCORE = "z_minscore"
COMPANY_NAME = "name"
COMPANY_NAME_CLEAN = "name_clean"
COMPANY_LOC = "full_address"
COMPANY_ALIAS = "alias"
COMPANY_ALIAS_CLEAN = "alias_clean"
COMPANY_ADDR1 = "address"
COMPANY_ADDR2 = "address2"
COMPANY_ADDR3 = "address3"
COMPANY_ADDR4 = "address4"
COMPANY_ADDR5 = "address5"
COMPANY_CITY = "city"
COMPANY_STATE = "state"
COMPANY_ZIP = "zip"
COMPANY_ZIP4 = "zip4"
COMPANY_CNTY = "county"
COMPANY_CNTRY = "country"
COMPANY_PROD = "product"
COMPANY_SIC1 = "sic"
COMPANY_SIC2 = "sic2"
COMPANY_SIC3 = "sic3"
COMPANY_SIC4 = "sic4"
COMPANY_NAIC = "naics"
COMPANY_DESC = "naicsdescr"
COMPANY_URL = "web"
COMPANY_TEL = "phone"
COMPANY_LAT = "latitude"
COMPANY_LON = "longitude"

# Zingg field names
CLUSTER_ID = COMPANY_Z_CLUSTER  # the column that specifies the cluster classification

# %%
# Group by cluster and then sort each by company name
grp = data[[CLUSTER_ID, COMPANY_NAME]].groupby(CLUSTER_ID)

grp.count().max()

"""NOTE
The result is that each z_cluster was a singlton.
So, the mapping function from Company to Facility is just the Identity function.
Hence, commenting out some blocks below . . .
"""
# %%
# Cluster and Company names
#
# c_names = grp.apply(
#     lambda x: reduce(lcs, x[COMPANY_NAME].sort_values(ascending=False).tolist())
# ).rename("name")
# c_names
#
# Review of `c_names` indicates that the `resolve_canonicals` functino is better than `lcs`.
# So we'll use `c_names2` as the cluster names, hence (:Company {name})-->(:Facility {COMPANY_NAME})

# %%
# Clusters and Canonical Company Names
# c_names = resolve_canonicals(data, name_col=COMPANY_NAME, cluster_col=COMPANY_Z_CLUSTER)
# c_names = (
#     pd.DataFrame.from_records(c_names, columns=[COMPANY_NAME, COMPANY_Z_CLUSTER])
#     .set_index(COMPANY_Z_CLUSTER)
#     .sort_index()
# )
# # Facility-to-Company Entity Mapping
# c_names

# # %% Save to file
# c_names.to_excel(LEVEL1_PATH / "c_names.xlsx")

c_names = data[[COMPANY_NAME]].drop_duplicates()

# %% COMPANY
# (:Company) node

companies = c_names.rename_axis(index=":ID(Company)")

companies

# obj_col = list(companies.select_dtypes(include='object'))
# for col in obj_col:
#     companies[col] = companies[col].apply(lambda x:x.replace('\n', ' ').replace('\r', ' '))

# %% Save
# companies[":LABEL"] = "Company"
# rather we specify the label at the CLI of `neo4j-admin import`
companies.to_csv(LEVEL2_PATH + "/Company.csv", index=True, sep=";")

# %% FACILITY
# (:Facility) node

facilities = c_names.rename_axis(index=":ID(Facility)")

facilities
# obj_col = list(facilities.select_dtypes(include='object'))
# for col in obj_col:
#     facilities[col] = facilities[col].apply(lambda x:x.replace('\n', ' ').replace('\r', ' '))
# %%

# facilities = (
#     data[COMPANY_NAME]
#     .sort_values()
#     .drop_duplicates()
#     .reset_index()
#     .drop(columns="index")
#     .rename_axis(index=":ID(Facility)")
# )
# %% Save
# facilities[":LABEL"] = "Facility"
facilities.to_csv(LEVEL2_PATH + "/Facility.csv", index=True, sep=";")

# %% FACILITY-to-COMPANY MAP


def lookup(df, col):
    lut = df[[col]].reset_index().set_index(col)

    def f(x):
        return lut.loc[x]

    return f


# %%
# Facility name-to-ID lookup table
f_lut = lookup(facilities, col=COMPANY_NAME)

# Company ID-to-name lookup table
c_lut = lookup(companies, col=COMPANY_NAME)

# %%
# (:Facility)-[:BELONGS_TO]->(:Company) edge

F_ID = ":ID(Facility)"
C_ID = ":ID(Company)"

data[F_ID] = data[COMPANY_NAME].apply(f_lut)[F_ID]
data[C_ID] = data[COMPANY_NAME].apply(c_lut)[C_ID]
belongs_to = data[[F_ID, C_ID]].drop_duplicates(keep="last")
belongs_to.rename(
    columns={F_ID: ":START_ID(Facility)", C_ID: ":END_ID(Company)"}, inplace=True
)

belongs_to

# obj_col = list(belongs_to.select_dtypes(include='object'))
# for col in obj_col:
#     belongs_to[col] = belongs_to[col].apply(lambda x:x.replace('\n', ' ').replace('\r', ' '))

# %%
# Save
belongs_to.to_csv(LEVEL2_PATH + "/BELONGS_TO.csv", index=False, sep=";")

# %% LOCATION
# ()-[:Location]->() edge

L_ID = ":ID(Location)"

LOCATION = [
    COMPANY_LOC,
    COMPANY_ADDR1,
    COMPANY_ADDR2,
    COMPANY_ADDR3,
    COMPANY_ADDR4,
    COMPANY_ADDR5,
    COMPANY_CITY,
    COMPANY_STATE,
    COMPANY_ZIP,
    COMPANY_CNTY,
    COMPANY_CNTRY,
]

locations = (
    data[LOCATION]
    .drop_duplicates(subset=COMPANY_LOC)
    .sort_values(
        by=[
            COMPANY_CNTRY,
            COMPANY_STATE,
            COMPANY_CNTY,
            COMPANY_ZIP,
            COMPANY_CITY,
            COMPANY_ADDR1,
            COMPANY_ADDR2,
        ]
    )
    .reset_index()
    .drop(columns="index")
    .rename_axis(L_ID)
)
assert locations.duplicated(subset=COMPANY_LOC).any() == False
locations
print('222')
obj_col = list(locations.select_dtypes(include='object'))
for col in obj_col:
    locations[col] = locations[col].apply(lambda x:x.replace('\n', ' ').replace('\r', ' '))

# %%Save
locations.to_csv(LEVEL2_PATH + "/Location.csv", index=True, sep=";")

print('abc')
# %%
l_lut = lookup(locations, col=COMPANY_LOC)
print('pqr')
data.to_csv(LEVEL2_PATH + "/data.csv", index=False, sep=";")
# %%
data[L_ID] = data[COMPANY_LOC].apply(l_lut)[L_ID]
print('lmn')
# %% LOCATED_IN relationship
# (:Facility)-[:LOCATED_IN]->(:Location) edge

LOCATED_IN = [
    F_ID,
    L_ID,
    COMPANY_URL,
    COMPANY_TEL,
    COMPANY_ZIP4,
    COMPANY_LAT,
    COMPANY_LON,
    COMPANY_PROD,
    COMPANY_NAIC,
    COMPANY_DESC,
    COMPANY_SIC1,
    COMPANY_SIC2,
    COMPANY_SIC3,
    COMPANY_SIC4,
]
print('sol')
located_in = (
    data[LOCATED_IN]
    .drop_duplicates(subset=[F_ID, L_ID], keep="last")
    .rename(
        columns={
            F_ID: ":START_ID(Facility)",
            L_ID: ":END_ID(Location)",
        }
    )
)
print('123')
# Better if URL is in all lowercase
located_in.loc[:, COMPANY_URL] = located_in[COMPANY_URL].str.lower()

print('567')

located_in
# %%
# Save to file
located_in.to_csv(LEVEL2_PATH + "/LOCATED_IN.csv", index=False, sep=";")
print('111')
# %%
# DONE
