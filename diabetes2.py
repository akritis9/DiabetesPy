#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:40:44 2021

@author: akritisharma
"""
# imports
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
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

# Create  classifier object. 
lr = LogisticRegression(C=1, penalty='l1',solver='liblinear')

# Create StratifiedKFold object. 
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 


for train_index, test_index in skf.split(x_df, y): 
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index] 
    lr.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold)) 
    

# Print the output. 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 
print('\nStandard Deviation is:', stdev(lst_accu_stratified)) 
    
    
#split into training and testing sets

# X_train, X_test, y_train, y_test = train_test_split(
#     df.drop(labels = 'Outcome', axis =1),
#     df['Outcome'],
#     test_size = 0.3,
#     random_state =1, stratify=df['Outcome'])

# X_train.shape, X_test.shape

# # scaler = StandardScaler()
# # scaler.fit(X_train.fillna(0))

# # selcting and fitting model with lasso
# sel_ = LogisticRegression(C=1, penalty='l1',solver='liblinear')
# model = sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

# # visualizing which features were deemed importnat or coeffs sent to zero
# model.coef_
# #since all true, all variables are included in the model


# y_pred = model.predict(X_test)

# test_mse = mean_squared_error(y_test, y_pred)

# # due to low test performance, try using k-fold cross val and
# # due to imbalanced outomes, try stratified sammpling when creating
# # test train splits
















