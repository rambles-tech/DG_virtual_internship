#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import sys

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
    #user_features = list(map(float, user_features))

    return user_features


@app.route('/result',methods = ['GET', 'POST'])
def result():

    print('This is error output', file=sys.stderr)
    # print('This is standard output', file=sys.stdout)   

    if request.method == 'POST':
            
        # vectorize user_input
        user_input = vectorize(request.form.to_dict())

        # predict
        result = ValuePredictor(user_input)

    else:
        
        s_length = request.args.get('s_length')
        s_width  = request.args.get('s_width')
        p_length = request.args.get('p_length')
        p_width  = request.args.get('p_width')

        
        x_predict = [s_length, s_width, p_length, p_width]
        result = ValuePredictor(x_predict)

    if result==0:
        prediction='Setosa'
    elif result==1:
        prediction='Versicolor'
    elif result==2:
        prediction='Virginica'
        
    return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)