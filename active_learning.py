

# Author: Hrad Ghoukasian
# Created: Sep 30, 2024
# License: MIT License
# Purpose: This python includes active learning with logistic regression 

# Usage: python active_learning.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added active learning implementation

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize an initial small labeled dataset
n_initial = 200
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size=n_initial, random_state=42)

# Define a simple classifier
def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Define uncertainty sampling strategy
def uncertainty_sampling(model, X_unlabeled, n_samples = 10):
    probas = model.predict_proba(X_unlabeled)
    uncertainty = 1 - np.max(probas, axis=1)  # Uncertainty is high when the probability of the most likely class is low
    return np.argsort(-uncertainty)[:n_samples]  # Select n_samples most uncertain samples

# Define diversity sampling strategy using K-Means clustering
def diversity_sampling(X_unlabeled, n_samples=10):
    kmeans = KMeans(n_clusters=n_samples, random_state=42)
    kmeans.fit(X_unlabeled)
    return np.argmin(cdist(X_unlabeled, kmeans.cluster_centers_, 'euclidean'), axis=1)

# Define representative sampling strategy
def representative_sampling(X_unlabeled, X_labeled, n_samples=10):
    distances = cdist(X_unlabeled, X_labeled, 'euclidean')
    return np.argmin(distances, axis=1)[:n_samples]

# Simulate active learning
sampling_methods = {
    "No Sampling": None,
    "Uncertainty Sampling": uncertainty_sampling,
    "Diversity Sampling": diversity_sampling,
    "Representative Sampling": representative_sampling,
}

active_learning_rounds = 15
results = {}
for method_name, sampling_fn in sampling_methods.items():
    X_current, y_current = X_labeled.copy(), y_labeled.copy()
    X_unlabeled_current, y_unlabeled_current = X_unlabeled.copy(), y_unlabeled.copy()  # Keep original data unchanged

    for _ in range(active_learning_rounds):  # Simulate "active_learning_rounds" rounds of active learning
        model = LogisticRegression()
        model.fit(X_current, y_current)

        if sampling_fn is not None:
            if method_name == "Uncertainty Sampling":
                indices = sampling_fn(model, X_unlabeled_current)
            elif method_name == "Representative Sampling":
                indices = sampling_fn(X_unlabeled_current, X_current)  # Pass X_current as X_labeled
            else:
                indices = sampling_fn(X_unlabeled_current)

            X_current = np.vstack([X_current, X_unlabeled_current[indices]])
            y_current = np.hstack([y_current, y_unlabeled_current[indices]])
            X_unlabeled_current = np.delete(X_unlabeled_current, indices, axis=0)
            y_unlabeled_current = np.delete(y_unlabeled_current, indices)

    accuracy, auc = train_and_evaluate(X_current, y_current, X_test, y_test)
    results[method_name] = (accuracy, auc)

# Display results
for method, (acc, auc) in results.items():
    print(f"{method} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# Optionally, plot the results
sampling_names = list(results.keys())
accuracies = [results[method][0] for method in sampling_names]
auc_scores = [results[method][1] for method in sampling_names]