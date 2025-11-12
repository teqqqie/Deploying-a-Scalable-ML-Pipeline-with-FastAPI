import pytest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.model import train_model, compute_model_metrics
from train_model import cat_features

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_model_algorithm_type():
    """
    Tests that the model is a RandomForestClassifier
    """

    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
def test_model_metrics():
    """
    Tests that the compute_model_metrics function returns the correct precision, recall, and F-beta scores.
    """
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    preds = model.predict(X)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert fbeta == fbeta_score(y, preds, beta=1, zero_division=1), "F-beta score does not match expected value"
    assert recall == recall_score(y, preds, zero_division=1), "Recall does not match expected value"
    assert precision == precision_score(y, preds, zero_division=1), "Precision does not match expected value"



# TODO: implement the third test. Change the function name and input as needed
def test_categorical_features_list():
    """
    Test that the list of categorical features passed in `train_model.py` is correct and contains the expected features.
    """
    # Your code here
    assert isinstance(cat_features, list), "cat_features in train_model.py should be a list"
    expected_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    for feature in expected_features:
        assert feature in cat_features, f"{feature} is missing from cat_features list in train_model.py"
