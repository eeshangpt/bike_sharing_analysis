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
# Eeshan Gupta  
# eeshangpt@gmail.com

# %% [markdown]
# # Table of Content
# 1. [Introduction to the problem](#Introduction-to-the-problem)
#     1. [Business Understanding](#Business-Understanding)
# 2. [Reading and cleaning the data](#Reading-and-Cleaning-the-data)
#     1. [Missing Values](#Missing-Data-Analysis)
#     2. [Cleaning the data](#Cleaning-the-data)
# 3. [Exploratory Data Analysis and Data Preparation](#Exploartory-Data-Analysis-and-Data-Preparation)
#     1. [Univariate Analysis](#)
#     2. [Segmented Univariate Analysis](#)
#     3. [Data Preparation](#)
#     4. [Bivariate Analysis](#)
# 4. [Data Preparation for Model Training](#Data-Preparations-for-Model-Training)
#     1. [Train Test Split](#Train-Test-Split)
#     2. [Rescaling](#Rescaling)
# 5. [Model Training](#Model-Training)
#     1. [Recursive Feature Elimination](#Recursive-Feature-Elimination)
#     2. [Manual Elimination of Features](#Manual-elimination-of-features)
# 6. [Residual Analysis](#Residual-Analysis)
# 7. [Predictions and Evaluation on Test Data](#Predictions-and-Evaluation-on-Test-Data)
# 8. [Subjective Questions](#Subjective-Questions)

# %% [markdown]
# ## Introduction to the problem
# ### Business Understanding
# - A US-based bike sharing provider **Boom Bikes**, provides bikes on sharing basis.
# - They have seen a dip in revenues due to the ongoing Corona Pandemic.
# - They want to understand the demands of the market when the quarantine situation ends allowing them an edge over the competitors
#
# **Business Goal**:
# - model the demand with the available independent variables
# - the client wants to understand how exactly the demands vary with different features
# - to manipulate the business strategy to meet the demands
# - understand the new business dynamics
#
# ### We intend to 
# - understand the factors on which the demands for these shared bikes depends in the American Market
# - know:
#     - Which variables are significant in predicting the demand for shared bikes
#     - How well those varibles describe the bikes demand

# %% [markdown]
# Imports and standard settings

# %%
import warnings
from os import getcwd
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    data_dictonary = f.readlines()
del f
print("".join(data_dictonary[:25]))

# %% [markdown]
# ### Missing Data Analysis

# %%
data.info()

# %%
data.isna().sum(axis=0)

# %% [markdown]
# The data has **NO** missing values in the provided data 

# %%
data.nunique()

# %% [markdown]
# The variable `instant` and `dteday` acts as indentifiers for the data. Rest all the columns neither have missing values nor have only one value for all the observstions. 

# %% [markdown]
# ### Cleaning the data

# %%
data['dteday'] = pd.to_datetime(data.dteday, format='%d-%m-%Y')
data.info()

# %% [markdown]
# Creating a copy of original data as a backup

# %%
data_bk = data.copy()

# %% [markdown]
# ## Exploartory Data Analysis and Data Preparation

# %% [markdown]
# Creating columns types.

# %%
target_columns = ["cnt"]

numerical_columns = ["temp",
                     "atemp",
                     "hum",
                     "windspeed",]

categorical_columns =["season",
                      "yr",
                      "mnth",
                      "holiday",
                      "weekday",
                      "workingday",
                      "weathersit",]

# %%
data.describe()

# %%
data[categorical_columns].sample(15)

# %% [markdown]
# ### Univariate Analysis

# %%
plt.figure()
data.season.value_counts().sort_index().plot(kind='bar')
plt.title('Seasons')
plt.show()

# %%
plt.figure()
data.yr.value_counts().sort_index().plot(kind='bar')
plt.title('Year')
plt.show()

# %%
plt.figure()
data.mnth.value_counts().sort_index().plot(kind='bar')
plt.title('Months')
plt.show()

# %%
plt.figure()
data.holiday.value_counts().sort_index().plot(kind='bar')
plt.title('Is Holiday?')
plt.show()

# %%
plt.figure()
data.weekday.value_counts().sort_index().plot(kind='bar')
plt.title('Weekday')
plt.show()

# %%
plt.figure()
data.workingday.value_counts().sort_index().plot(kind='bar')
plt.title('Is Working Day?')
plt.show()

# %%
plt.figure()
data.weathersit.value_counts().sort_index().plot(kind='bar')
plt.title('Weather')
plt.show()

# %%
plt.figure()
sns.boxplot(data[['temp', 'atemp']]) #, 'hum', 'windspeed',]])
plt.title('Temperatures (Actual and Adjusted)')
plt.show()

# %%
plt.figure()
sns.boxplot(data.hum)
plt.title('Humidity')
plt.show()

# %%
plt.figure()
sns.boxplot(data.windspeed)
plt.title('Windspeed')
plt.show()

# %%
plt.figure()
sns.boxplot(data.cnt)
plt.title("Count of total Customers")
plt.show()

# %% [markdown]
# ### Segmented-univariate Analysis

# %%
sns.pairplot(data[numerical_columns + target_columns])
plt.show()

# %% [markdown]
# We can infer that 

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
    sns.boxplot(data=data, x=col, y='cnt')

# %% [markdown]
# ### Data Preparation

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
data.head(10)

# %%
predictor_columns = [col for col in data.columns if col not in target_columns]

# %% [markdown]
# ### Bivariate Analysis

# %%
plt.figure(figsize=(20, 8))
sns.heatmap(data[[i for i in data.columns 
                  if i not in target_columns]].corr().round(1), annot=True)
plt.show()

# %%
plt.figure(figsize=(20, 8))
sns.heatmap(data[[i for i in data.columns]].corr().round(1), annot=True)
plt.show()

# %% [markdown]
# **EDA Summary**
# - There seems to be a slightly more number of observations for seasons `fall (3)` followed by `summer (2)`
# - Weather seems to `1: Clear, Few clouds, Partly cloudy, Partly cloudy` most of the times  
#         - It also implies that more customer used the service on a clear day  
# - The Feeling temperature `atemp` is generally higher than the actual temperature `temp`
#         - There is a linear relation between them
# - Median number of customers are higher in the `fall (3)` season followed by `summer (2)` and then `winter (4)`. `Sprin
# g (1)` has the lowest amount of customers 
# - The number of customers increased in 2019 as compared to 2018     
# - Customer used the service more on a day which was not a `holiday`, although this is not true for weekends `weekday (0)`

# %% [markdown]
# ## Data Preparations for Model Training

# %% [markdown]
# ### Train Test Split

# %%
data_train, data_test = train_test_split(data, train_size=0.7, random_state=0)

# %%
data_train.info()

# %%
data_test.info()

# %% [markdown]
# ### Rescaling

# %%
min_max_scalar = MinMaxScaler()

# %%
data_train[numerical_columns + target_columns] = min_max_scalar.fit_transform(data_train[numerical_columns + target_columns])
data_train[numerical_columns + target_columns].describe()

# %%
data_test[numerical_columns + target_columns] = min_max_scalar.transform(data_test[numerical_columns + target_columns])
data_test[numerical_columns + target_columns].describe()

# %%
train_y = data_train.pop('cnt')
train_X = data_train

# %%
test_y = data_test.pop('cnt')
test_X = data_test

# %% [markdown]
# ## Model Training

# %% [markdown]
# ### Recursive Feature Elimination
#
# Total number of features in the data are **30**. I am choosing to eliminate **50%** of the feature and moving ahead with **15** features.

# %%
lr_model = LinearRegression()
lr_model.fit(train_X, train_y)

rfe = RFE(lr_model, n_features_to_select=15)
rfe.fit(train_X, train_y)

# %%
pd.DataFrame(
    list(zip(train_X.columns, rfe.support_, rfe.ranking_)),
    columns=['Feature', 'Selected?', 'Rank']).sort_values(by=['Rank', 'Feature']).reset_index(drop=True)

# %% [markdown]
# Selected Features after RFE

# %%
selected_columns = train_X.columns[rfe.support_]
pd.Series(selected_columns.to_list())

# %% [markdown]
# Dropped Features after RFE

# %%
dropped_columns = train_X.columns[~rfe.support_]
pd.Series(dropped_columns.to_list())

# %% [markdown]
# Filtering the selected columns

# %%
train_X_rfe = train_X[selected_columns]

# %% [markdown]
# ### Manual Elimination of Features

# %% [markdown]
# #### Step 1

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

# %% [markdown]
# #### Step 2
# Since we are getting very high values of VIF ($\infty$), I am choosing to eliminate 2 variables `weekday_3`, & `weekday_5` based on P-values which are $0.915$ & $0.833$ respectively

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

# %% [markdown]
# #### Step 3
#
# Although the VIF values are high ($\gt 5$) for `hum` and `atemp`, but their P-values are $0$. Instead I am choosing to drop a variable with very high P-value or about which the model is not sure about. In this case it is `weekday_4` with P-value $0.789$

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

# %% [markdown]
# #### Step 4
#
# Again we observe that `hum` and `atemp` have a high VIF ($\gt 5$) but $0$ P-Value. Again I am choosing to drop a variable with very high P-value (or about which the model is not sure about). In this case it is `weekday_2` with P-value $0.351$

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

# %% [markdown]
# #### Step 5
#
# Again we observe that `hum` and `atemp` have a high VIF ($\gt 5$) but $0$ P-Value. Again I am choosing to drop a variable with very high P-value (or about which the model is not sure about). In this case it is `workingday` with P-value $0.337$

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

# %% [markdown]
# #### Step 6
#
# Again we observe that `hum` and `atemp` have a high VIF ($\gt 5$) but $0$ P-Value. Again I am choosing to drop a variable with very high P-value (or about which the model is not sure about). In this case it is `weekday_1` with P-value $0.309$

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

# %% [markdown]
# Removing further fetures such as those with high will hurt $R^2$ value. Therefore I am choosing to keep all the features used for last training and finalizing the model.

# %% [markdown]
# ## Residual Analysis

# %% [markdown]
# Generating Predictions for the training data

# %%
train_y_pred = lr_sm_model.predict(train_X_rfe)
train_y_pred

# %%
plt.figure()
sns.histplot((train_y_pred - train_y), bins=100, kde=True)
plt.xlabel('Residuals')
plt.ylabel('')
plt.title('')
plt.show()

# %% [markdown]
# We observe that residuals are centered at $0$ and are normally distributed. 

# %% [markdown]
# ## Predictions and Evaluation on Test Data

# %%
test_X_rfe = sm.add_constant(test_X[[i for i in train_X_rfe.columns if i != 'const']])
test_X_rfe.head()

# %%
test_y_pred = lr_sm_model.predict(test_X_rfe)

# %%
r2_score(test_y, test_y_pred)

# %%
mean_squared_error(test_y, test_y_pred)

# %% [markdown]
# ## Subjective Questions

# %% [markdown]
# ### Question 1
#
# | variable | $\beta_i$ value|
# |--------------|:-----:|
# |const|0.1962|
# |yr|0.2161|
# |holiday|-0.0603|
# |workingday|0.0066|
# |atemp|0.6006|
# |hum|-0.2122|
# |windspeed|-0.1281|
# |season_2|0.0822|
# |season_4|0.1287|
# |mnth_9|0.1102|
# |weathersit_3|-0.1475|
#
# We can infer that:
# - On windy days, the number of customers will be less
# - Same can be said about a holiday
# - On high humidity days the number of customers will be even less.
#   with increare in temperature, the number of customers will increase
#

# %% [markdown]
# ### Question 2
# Using `drop_first=True` creates $k - 1$ dummy columns where $k$ is the number of categories. This allow us to use less columns to repesent the same information. This can be thought of as following. Suppose we variable `sex` which has 2 values `M` and `F`. Now if the column `is_male` containing `0` and `1` can describe the variable completely, and we do not require `is_female` columns to capture the all the information of the `sex` column.

# %% [markdown]
# ### Question 3
# `temp` and `atemp` are the 2 highest correlated to the target variable

# %% [markdown]
# ### Question 4
#
# We can validate by checking the distribution of the residuals:
# - The errors should be random
# - The errors should be centered around 0
#
# Also the VIF for all the selected variables is less than 10

# %% [markdown]
# ### Question 5
# The top-3 features that contibute significantly towards expalining the demand are:
# 1. `yr`
# 2. `atemp`
# 3. `season_4`
