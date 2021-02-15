#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:00:43 2021

@author: akritisharma
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('rf_final.pkl','rb')
rf_final = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    Pregnancies = request.args.get('Pregnancies')
    Glucose = request.args.get('Glucose')
    BloodPressure = request.args.get('BloodPressure')
    SkinThickness = request.args.get('SkinThickness')
    Insulin = request.args.get('Insulin')
    BMI = request.args.get('BMI')
    DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')
    Age = request.args.get('Age')
    
    prediction = rf_final.predict([[Pregnancies,Glucose,BloodPressure,
                                   SkinThickness,Insulin, BMI, 
                                   DiabetesPedigreeFunction, Age]])
    
    return "The predicted value is " + str(prediction)






if __name__ == '__main__':
    app.run()