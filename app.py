import numpy as np
from flask import Flask,request, render_template
import pickle

#__name__ is the name of the current Python module. The app needs to know where it’s located to set up some paths, and __name__ is a convenient way to tell that.It’s used by Flask to identify resources like templates, static files.

app = Flask(__name__) # create Flask instance
model = pickle.load(open('model.pkl', 'rb')) #to open the model.. in read mode

#route() binds the url to a function
@app.route('/')   # default route.. when '/' is in url, home() will be called & 'index.html' will be returned
def home():
    return render_template('index.html')
# the o/p of home() will be shown in the browser

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Sales  $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)