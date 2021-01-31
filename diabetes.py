#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:40:44 2021

@author: akritisharma
"""
# imports
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# load data
df = pd.read_csv("diabetes.csv", low_memory = False)

# view vriables
varlist = list(df.columns)
# ['Pregnancies',
#  'Glucose',
#  'BloodPressure',
#  'SkinThickness',
#  'Insulin',
#  'BMI',
#  'DiabetesPedigreeFunction',
#  'Age',
#  'Outcome']

# set Outcome as the dependent variable

y = df['Outcome']
