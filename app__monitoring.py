from flask import Flask, render_template, request
import pickle
import pandas as pd
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
from dotenv import load_dotenv
import os

import datetime

# Charger les variables d'environnement
load_dotenv()

ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

# Initialisation du client Arize
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Définition du schéma de données pour Arize
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding", "income", "years_employed", "fico_score"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)

# Initialisation de l'application Flask et du modèle de régression logistique
app = Flask(__name__)
model = pickle.load(open("model(2).pkl", "rb"))  # Chargement du modèle de régression logistique

# Fonction de prédiction
def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])

# Page d'accueil
@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

# Prédiction pour le défaut de paiement
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
        total_debt_outstanding = float(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score = int(request.form["fico_score"])
        actual_label = int(request.form.get("actual_label", 0))  # Si disponible, sinon 0

        # Features utilisées pour la prédiction
        features = [credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]
        
        # Effectuer la prédiction
        prediction = model_pred(features)

        # Enregistrer la prédiction dans Arize
        timestamp = pd.Timestamp.now()
        data = {
            "prediction_id": [str(timestamp.timestamp())],  # ID unique pour chaque prédiction
            "timestamp": [timestamp],
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
            "prediction_label": [prediction],
            "actual_label": [actual_label]  # Label réel pour l'évaluation
        }
        dataframe = pd.DataFrame(data)
        
        try:
            # Enregistrer les données dans Arize
            response = arize_client.log(
                dataframe=dataframe,
                model_id="model(2)",
                model_version="v1",
                model_type=ModelTypes.SCORE_CATEGORICAL,  # Type de modèle de régression logistique (binaire)
                environment=Environments.PRODUCTION,
                schema=schema
            )
            if response.status_code != 200:
                print(f"Échec de l'enregistrement dans Arize: {response.text}")
            else:
                print("Prédiction enregistrée avec succès dans Arize")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement dans Arize: {e}")
        
        # Message à afficher en fonction de la prédiction
        if prediction == 1:
            return render_template("index.html", prediction_text="Il y a un risque de défaut de paiement. Veuillez contacter un conseiller.")
        else:
            return render_template("index.html", prediction_text="Pas de risque de défaut de paiement.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 
