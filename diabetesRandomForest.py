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
rf_model = RandomForestClassifier(random_state=40)

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


    
#train model using entire dataset to prep for deployment

rf_final = RandomForestClassifier()

rf_final.fit(x_scaled,y)

#rf_finalFit.decision_path(x_scaled)
#since model has been trained on scaled features, 

#plot variable importance
# credit: analyseup.com
def plot_feature_importance(importance,names,model_type):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

# call plot function

plot_feature_importance(rf_final.feature_importances_,x_df.columns,'Random Forest')

#create pickle file using serialization
import pickle
pickle_out = open("rf_final.pkl","wb")
pickle.dump(rf_final, pickle_out)
pickle_out.close()





# remember to scale user input