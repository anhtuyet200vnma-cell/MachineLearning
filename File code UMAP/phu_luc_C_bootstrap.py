import numpy as np
import pandas as pd

from common import (
    load_data,
    split_data,
    standardize_data,
    reduce_with_umap,
    train_logistic_regression
)

from sklearn.metrics import accuracy_score


# Load and split data
X, y, target_names = load_data()

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled, scaler = standardize_data(
    X_train, X_test
)

# Train UMAP 2D
X_train_umap, X_test_umap, reducer = reduce_with_umap(
    X_train_scaled,
    X_test_scaled,
    n_components=2
)

# Train model
model = train_logistic_regression(
    X_train_umap,
    y_train
)

# Predict test data
predictions = model.predict(X_test_umap)

# Bootstrap
rng = np.random.default_rng(42)

n_bootstrap = 100
bootstrap_scores = []

for i in range(n_bootstrap):

    indices = rng.integers(
        0, len(y_test),
        size=len(y_test)
    )

    y_sample = y_test[indices]
    pred_sample = predictions[indices]

    score = accuracy_score(
        y_sample,
        pred_sample
    )

    bootstrap_scores.append(score)

# Results
mean_score = np.mean(bootstrap_scores)
lower = np.percentile(bootstrap_scores, 2.5)
upper = np.percentile(bootstrap_scores, 97.5)

print("===== BOOTSTRAP RESULTS =====")
print("Number of Bootstrap samples:", n_bootstrap)
print("Mean accuracy:", round(mean_score, 4))
print("95% interval:", round(lower, 4), "-", round(upper, 4))

pd.DataFrame({
    "Bootstrap Accuracy": bootstrap_scores
}).to_csv("bootstrap_results.csv", index=False)