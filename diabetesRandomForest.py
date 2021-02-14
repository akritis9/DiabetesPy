#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:06:16 2021

@author: akritisharma
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev 
from sklearn import preprocessing 
from sklearn import datasets 



from matplotlib import pyplot as plt

# load data
df = pd.read_csv("diabetes.csv", low_memory = False)

df.shape

# view vriables
varlist = list(df.columns)
# ['Pregnancies': Number of times pregnant,
#  'Glucose':Plasma glucose concentration a 2 hours in an oral glucose tolerance test,
#  'BloodPressure': Diastolic blood pressure (mm Hg),
#  'SkinThickness': Triceps skin fold thickness (mm),
#  'Insulin' : 2-Hour serum insulin (mu U/ml),
#  'BMI' : Body mass index (weight in kg/(height in m)^2),
#  'DiabetesPedigreeFunction': Diabetes pedigree function(a function which scores likelihood of diabetes based on family history),
#  'Age' :Age (years),
#  'Outcome' : Class variable (0 or 1) 268 of 768 are 1, the others are 0 (non)]

# set Outcome as the dependent variable

y = df['Outcome']

# make sure our data is clean- i.e it has no null values

df.info()

corr = df.corr()
print(corr)

sns.heatmap(corr,
            xticklabels = corr.columns,
            yticklabels = corr.columns)

#create datasets

y = df['Outcome']

x_df = df.drop(labels = 'Outcome', axis =1)

# Feature Scaling for input features. 
scaler = preprocessing.MinMaxScaler() 
x_scaled = scaler.fit_transform(x_df) 

# create a classifier object
rf_model = RandomForestClassifier()

# Create StratifiedKFold object. 
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 


for train_index, test_index in skf.split(x_df, y): 
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index] 
    rf_model.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(rf_model.score(x_test_fold, y_test_fold)) 

# Print the output. 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 
print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 
    
