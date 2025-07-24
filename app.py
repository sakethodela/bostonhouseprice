import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,app,jsonify,url_for,render_template
#create a app using flask framework
app = Flask(__name__)
#load model using pickle file
regmodel=pickle.load(open('regression_model.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
#route to web i.e., after localhost /
@app.route('/')
#we need to route to home page on web so, create a method
def home():
    return render_template("home.html")
#create an api becuase using api calls our model predict output
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data'] #request for data which in json format key&value pair
    print(data)
    #convert list to values, convert to array, reshape because in our regression problem data in 2-D, so, we are converting them.
    print(np.array(list(data.values())).reshape(1,-1))
    #we collet data need to standardze so, for each fetaure record it produces mutliple datapoints as rows so, we are converting them into features as final for each record has multiple datapoints as features.
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    #predict the data using our model
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)
    return render_template("home.html",prediction_text="the house price prediction is {}".format(output))
# runnng our created name app by making conditions like euqls to main app
if __name__ == "__main__":
    app.run(debug=True)