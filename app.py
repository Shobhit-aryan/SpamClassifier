# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:18:30 2020

@author: Asus
"""
from flask import Flask, render_template, request
import pickle


classifier=pickle.load(open('modelPickle.pkl','rb'))
cv=pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        
        if my_prediction==1:
            return render_template('index.html',prediction_text="Ohh! This is a spam message")
        else:
            return render_template('index.html',prediction_text="Don't Worry! This is not a spam message")
        
    
if __name__=='__main__':
        app.run(debug=True)