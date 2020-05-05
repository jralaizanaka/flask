from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import numpy as np
import requests
import json

import pandas as pd

app = Flask(__name__)

# Chargement des fichiers avec les objets sklearn pour le preprocessing et le modèle
imputer = joblib.load("imputer.pkl")
featureencoder = joblib.load("featureencoder.pkl")
labelencoder = joblib.load("labelencoder.pkl")
classifier = joblib.load("classification_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        # Recover informations from html form
        data = dict(request.form.items())

        country = data["Country"]
        # handling missing fields for age and salary
        try:
            age = float(data["Age"])
        except ValueError:
            age = None
        try:
            salary = float(data["Salary"])
        except ValueError:
            salary = None

        # Create DataFrame with columns in the same order as in src/Data.csv

        d = {'Country': [country], 'Age': [age],'Salary': [salary]}
        X = pd.DataFrame(data=d)

        # Convert dataframe to numpy array before using scikit-learn

        X=X.values

        # Preprocessings : impute and scale/encode features

        numeric_indices=[1,2]
        categorical_indices=[0]
        X[:,numeric_indices] = imputer.transform(X[:,numeric_indices])
        X = featureencoder.transform(X)

        # Prediction

        prediction = classifier.predict(X)

        # Use labelencoder to translate prediction into 'yes' or 'no'

        prediction_translated=labelencoder.inverse_transform(prediction)


    return render_template("predicted.html", text=prediction_translated)


if __name__ == '__main__':
    app.run(debug=True)
