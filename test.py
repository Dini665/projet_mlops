from app import model_pred

new_data = {
    'credit_lines_outstanding': 0,
    'loan_amt_outstanding': 5221.545193,
    'total_debt_outstanding': 3915.471226,
    'income': 78039.38546,
    'years_employed': 5,
    'fico_score': 605
}

def test_predict():
    prediction = model_pred(new_data)
    assert prediction in [0, 1], "Erreur de prédiction"