#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4) # 4 attributes
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return int(result[0])

def vectorize(user_input):
    user_features = user_input
    user_features = list(user_features.values())
    user_features = list(map(float, user_features))

    return user_features


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        
        # vectorize user_input
        user_input = vectorize(request.form.to_dict())

        # predict
        result = ValuePredictor(user_input)

        if result==0:
            prediction='Setosa'
        elif result==1:
            prediction='Versicolor'
        elif result==2:
            prediction='Virginica'
            
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
	app.run(debug=False)