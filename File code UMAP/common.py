import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import umap


def load_data():
    """Load Breast Cancer Wisconsin dataset."""
    dataset = load_breast_cancer()
    return dataset.data, dataset.target, dataset.target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def standardize_data(X_train, X_test):
    """Standardize data without fitting on test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calculate test accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions


def reduce_with_umap(X_train, X_test, n_components=2):
    """Reduce dimensions using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=n_components,
        metric="euclidean",
        random_state=42
    )

    X_train_umap = reducer.fit_transform(X_train)
    X_test_umap = reducer.transform(X_test)

    return X_train_umap, X_test_umap, reducer