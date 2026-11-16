import pandas as pd
import umap

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# Baseline: 30 original features
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

scores = cross_val_score(
    baseline, X, y,
    cv=cv,
    scoring="accuracy"
)

results.append({
    "Method": "Original",
    "Dimensions": 30,
    "Mean Accuracy": scores.mean(),
    "Std": scores.std()
})

# UMAP: 2, 5, 10 dimensions
for n in [2, 5, 10]:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("umap", umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=n,
            random_state=42
        )),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "Method": "UMAP",
        "Dimensions": n,
        "Mean Accuracy": scores.mean(),
        "Std": scores.std()
    })

# Display results
table = pd.DataFrame(results)

print("===== 5-FOLD CROSS VALIDATION =====")
print(table.to_string(index=False))

table.to_csv(
    "cross_validation_results.csv",
    index=False
)