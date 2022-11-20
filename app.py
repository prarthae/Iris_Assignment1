import pickle
import numpy as np
from flask import Flask , render_template,request

model = pickle.load(open("iris_dt_model.pkl","rb"))

app = Flask(__name__)

@app.route("/")

def home():
    return render_template("index.html")


@app.route("/predict",methods = ['POST'])
def predict():

        init_features = [float(x) for x in request.form.values()]
        final_features = [np.array(init_features)]
        prediction= model.predict(final_features)

        return render_template("index.html",prediction_text = "Precited class: {}".format(prediction))

if __name__== "_main_":
    app.run(debug=True)