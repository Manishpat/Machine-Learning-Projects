import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Crop production will be  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)