#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:50:06 2021

@author: akritisharma
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)



pickle_in = open('rf_final.pkl','rb')
rf_final = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict', methods =["Get"])
def predict_note_authentication():
    
    """Let's authenticate the diabetes prediction
    This is using docstrings for specifications.
    ---
    parameters:
        - name: Pregnancies
          in: query
          type: number
          required: true
        - name: Glucose
          in: query
          type: number
          required: true
        - name: BloodPressure
          in: query
          type: number
          required: true
        - name: SkinThickness
          in: query
          type: number
          required: true
        - name: Insulin
          in: query
          type: number
          required: true
        - name: BMI
          in: query
          type: number
          required: true
        - name: DiabetesPedigreeFunction
          in: query
          type: number
          required: true
        - name: Age
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    
    """
        
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
    print(prediction)
    
    return "The predicted value is " + str(prediction)






if __name__ == '__main__':
    app.run()