# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Regression Analysis
#
# Analysis of a dataset by a US bike-sharing provider **BoomBikes**.
# The client **BoomBikes**: 
# - has suffered considerable dip in revenues due to the ongoing Corona pandic
# - wants undestand the demand for shared bikes market after the quarantine situation ends
# - this will allow them to preapre for the market need once lockdowns ends and stand out from the comptetion
#
# Our Task:
# - understand the factors on which the demands for these shared bikes depends
# - in the American Market
# - wants to know:
#     - which variables are significant in predicting the demand for shared bikes
#     - how well those varibles describe the bikes demand
#
# **Business Goal**:
# - model the demand with the available independent variables
# - the client wants to understand how exactly the demands vary with different features
# - to manipulate the business strategy to meet the demands
# - understand the new business dynamics

# %% [markdown]
# # TABLE OF CONTENT
# 1. [Introduction to the problem](#Introduction-to-the-problem)
# 2. [Motivation](#Motivation)
# 3. [Reading and cleaning the data]()
# 4. [Univariate and Segmented Univariate Analysis]()
# 5. [Bivariate Analysis]()
# 6. [Segmenting the data]()

# %% [markdown]
# ## Introduction to the problem

# %% [markdown]
# ## Motivation

# %%
import warnings
from os import getcwd
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
np.random.seed(0)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', None)
sns.set_style('darkgrid')

# %%
PRJ_DIR = getcwd()
DATA_DIR = join(PRJ_DIR, "data")

# %% [markdown]
# ## Reading and Cleaning the data

# %%
data = pd.read_csv(join(DATA_DIR, "day.csv"))
data.head()

# %%
with open(join(DATA_DIR, "Readme.txt"), 'r') as f:
    data_dictonary = f.read()

print(data_dictonary)

# %% [markdown]
# ### Missing Data Analysis

# %%
data.info()

# %% [markdown]
# The data has no null values. 

# %%
data.nunique()

# %%
data.isna().sum(axis=0)

# %% [markdown]
# ### Cleaning the data

# %%
data['dteday'] = pd.to_datetime(data.dteday, format='%d-%m-%Y')

# %%
data.info()

# %%
data.nunique()

# %%
data_bk = data.copy()

# %% [markdown]
# ## Univariate and Segemented Univariate Analysis

# %%
target_columns = [
# "casual",
# "registered",
"cnt",
]

# %%
numerical_columns = [
"temp",
"atemp",
"hum",
"windspeed",
]

# %%
categorical_columns =[
"season",
"yr",
"mnth",
"holiday",
"weekday",
"workingday",
"weathersit",
]

# %%
data.describe()

# %%
data[categorical_columns]

# %%
categorical_col_gt_2_unique_values = [i for i in categorical_columns if data[i].nunique() > 2]
dummy_col_df = pd.DataFrame()
for i in categorical_col_gt_2_unique_values:
    a = pd.get_dummies(data[i],prefix=f'{i}', prefix_sep='_', drop_first=True).astype(int)
    dummy_col_df = pd.concat([dummy_col_df, a], axis=1)
    del a

# %%
data = pd.concat([data, dummy_col_df], axis=1)
data.head()

# %%
columns_to_dropped = list(set(categorical_col_gt_2_unique_values).intersection(set(categorical_columns)))
del categorical_col_gt_2_unique_values, categorical_columns, dummy_col_df

# %%
data = data.drop(columns_to_dropped + ['dteday', 'instant', 'casual', 'registered'], axis=1)

# %%
data

# %%
predictor_columns = [col for col in data.columns if col not in target_columns]

# %%
sns.pairplot(data[numerical_columns + target_columns])
plt.show()

# %%
categorical_columns = [i for i in data_bk.columns 
                       if i not in numerical_columns + target_columns + ['instant', 'dteday', 'casual', 'registered']]
categorical_columns

# %%
(data.cnt < 0).sum()

# %%
plt.figure(figsize=(15, 20))
for i, col in enumerate(categorical_columns):
    plt.subplot(4, 2, i + 1)
    sns.violinplot(data=data_bk, x=col, y='cnt')

# %%
plt.figure(figsize=(20, 20))
sns.heatmap(data[[i for i in data.columns 
                  if i not in target_columns]].corr().round(2), annot=True)
plt.show()

# %%
data[target_columns].describe()

# %%
plt.figure()
sns.boxplot(data[target_columns])
plt.show()

# %%
data_train, data_test = train_test_split(data, train_size=0.7, random_state=0)

# %%
data_train.info()

# %%
data_test.info()

# %%
min_max_scalar = MinMaxScaler()

# %%
data_train[numerical_columns + target_columns] = min_max_scalar.fit_transform(data_train[numerical_columns + target_columns])

# %%
data_train[numerical_columns + target_columns].describe()

# %%
train_y = data_train.pop('cnt')
train_X = data_train

# %%
lr_model = LinearRegression()
lr_model.fit(train_X, train_y)

rfe = RFE(lr_model, n_features_to_select=15)
rfe.fit(train_X, train_y)

# %%
pd.DataFrame(
    list(zip(train_X.columns, rfe.support_, rfe.ranking_)),
    columns=['Feature', 'Selected?', 'Rank']).sort_values(by=['Rank', 'Feature']).reset_index(drop=True)

# %%
selected_columns = train_X.columns[rfe.support_]
selected_columns.to_list()

# %%
dropped_columns = train_X.columns[~rfe.support_]
dropped_columns.to_list()

# %%
train_X_rfe = train_X[selected_columns]

# %%
train_X_rfe = sm.add_constant(train_X_rfe)
train_X_rfe.head()

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by=['VIF'], ascending=False)

# %%

# %%
train_X_rfe = train_X_rfe.drop(['weekday_3', 'weekday_5'], axis=1)

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by=['VIF'], ascending=False)

# %%

# %%
train_X_rfe = train_X_rfe.drop('weekday_4', axis=1)

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by=['VIF'], ascending=False)

# %%
train_X_rfe = train_X_rfe.drop(['weekday_2'], axis=1)

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by='VIF', ascending=False).reset_index(drop=True)

# %%

# %%
train_X_rfe = train_X_rfe.drop(['workingday'], axis=1)

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by='VIF', ascending=False)

# %%

# %%
train_X_rfe = train_X_rfe.drop(['weekday_1'], axis=1)

# %%
lr_sm_model = sm.OLS(train_y, train_X_rfe).fit()
lr_sm_model.summary()

# %%
temp_X = train_X_rfe.drop('const', axis=1)
pd.DataFrame([
    (i, round(variance_inflation_factor(temp_X.values, itr), 2)) 
    for itr, i in enumerate(temp_X.columns)],
             columns=['Features', 'VIF']).sort_values(by='VIF', ascending=False).reset_index(drop=True)

# %%
data_bk.groupby('yr')['cnt'].sum().plot(kind='bar')

# %%
